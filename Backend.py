"""FieldSense - AI-based Sports Performance Tracker (FastAPI backend, single file).
 
Sections (split into modules later): config, database, models, auth, utilities,
AI/ML, session engine, demo generator, WebSocket hub, API routes.
Run:  uvicorn backend:app --reload --host 0.0.0.0 --port 8000
"""
import asyncio, math, os, random, re
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional
 
import bcrypt, jwt
from bson import ObjectId
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from pymongo.errors import PyMongoError
 
# ----------------------------------------------------------------- config
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "sports_tracker")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "12"))
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",")]
DEVICE_KEY = os.getenv("DEVICE_KEY")  # optional shared secret the ESP32 sends as X-Device-Key
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")  # optional trained Random Forest
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET is not set. Add it to your .env file.")
 
client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=3000, tz_aware=True)
db = client[DB_NAME]  # collections: users, sessions, sensor_readings, devices
EMAIL_RE = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
DEVICE_TIMEOUT_S = 15
 
 
# ----------------------------------------------------------------- models
class Signup(BaseModel):
    full_name: str = Field(min_length=2, max_length=80)
    email: str = Field(pattern=EMAIL_RE, max_length=120)
    password: str = Field(min_length=8, max_length=72)
    age: int = Field(ge=10, le=100)
    gender: str = Field(max_length=20)
    height_cm: float = Field(ge=100, le=250)
    weight_kg: float = Field(ge=25, le=250)
    sport: str = Field(max_length=40)
    fitness_level: str = "Beginner"
 
 
class Login(BaseModel):
    email: str
    password: str
 
 
class ProfileUpdate(BaseModel):
    full_name: Optional[str] = Field(None, min_length=2, max_length=80)
    age: Optional[int] = Field(None, ge=10, le=100)
    gender: Optional[str] = Field(None, max_length=20)
    height_cm: Optional[float] = Field(None, ge=100, le=250)
    weight_kg: Optional[float] = Field(None, ge=25, le=250)
    sport: Optional[str] = Field(None, max_length=40)
    fitness_level: Optional[str] = Field(None, max_length=20)
 
 
class DeviceReading(BaseModel):
    device_id: str = Field(min_length=1, max_length=40)
    heart_rate: Optional[float] = Field(None, ge=30, le=230)
    temperature: Optional[float] = Field(None, ge=-20, le=70)
    humidity: Optional[float] = Field(None, ge=0, le=100)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    speed: Optional[float] = Field(None, ge=0, le=80, description="km/h")
 
 
class StartReq(BaseModel):
    demo: bool = False
 
 
class DeviceReg(BaseModel):
    device_id: str = Field(min_length=1, max_length=40)
 
 
# ----------------------------------------------------------------- utilities
def now():
    return datetime.now(timezone.utc)
 
 
def clean(o):
    """Make Mongo documents JSON-safe and strip secrets."""
    if isinstance(o, dict):
        return {("id" if k == "_id" else k): clean(v) for k, v in o.items() if k not in ("password_hash", "last_point")}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, ObjectId):
        return str(o)
    if isinstance(o, datetime):
        return o.isoformat()
    return o
 
 
def oid(value):
    try:
        return ObjectId(value)
    except Exception:
        raise HTTPException(404, "Not found")
 
 
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))
 
 
def elapsed(s, at=None):
    end = s.get("ended_at") or at or now()
    if s["status"] == "paused" and s.get("paused_at"):
        end = s["paused_at"]
    return max(0.0, (end - s["started_at"]).total_seconds() - s.get("paused_total", 0))
 
 
MET_TABLE = [(2, 2.0), (4, 3.0), (6.5, 4.5), (8, 8.0), (10, 9.8), (12, 11.0)]
 
 
def met_for(speed):
    for limit, met in MET_TABLE:
        if speed < limit:
            return met
    return 12.5
 
 
# ----------------------------------------------------------------- auth
bearer = HTTPBearer(auto_error=False)
 
 
def hash_pw(pw):
    return bcrypt.hashpw(pw.encode()[:72], bcrypt.gensalt()).decode()
 
 
def check_pw(pw, hashed):
    try:
        return bcrypt.checkpw(pw.encode()[:72], hashed.encode())
    except ValueError:
        return False
 
 
def make_token(uid):
    return jwt.encode({"sub": str(uid), "exp": now() + timedelta(hours=JWT_HOURS)}, JWT_SECRET, algorithm="HS256")
 
 
def decode_token(token):
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Session expired. Please log in again.")
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid token")
 
 
async def current_user(cred: Optional[HTTPAuthorizationCredentials] = Depends(bearer)):
    if not cred:
        raise HTTPException(401, "Not authenticated")
    user = await db.users.find_one({"_id": oid(decode_token(cred.credentials))})
    if not user:
        raise HTTPException(401, "Account not found")
    return user
 
 
# ----------------------------------------------------------------- AI / ML
# Features: age, avg_hr, max_hr, avg_speed, max_speed, distance_km, duration_min, calories, temp, humidity
# NOTE: no real labelled dataset exists yet. The score is a transparent rule-based baseline and the
# Random Forest below is trained on SYNTHETIC samples labelled by that baseline. To use real data, train
# your own RandomForestClassifier on the same feature order (labels 0..3) and save it as model.joblib.
CATS = ["Needs Improvement", "Average", "Good", "Excellent"]
MODEL, MODEL_NAME = None, "rule-baseline"
 
 
def category_idx(score):
    return 3 if score >= 80 else 2 if score >= 65 else 1 if score >= 45 else 0
 
 
def baseline_score(f):
    age, avg_hr, _mx, avg_sp, _msp, _d, dur, _cal, temp, hum = f
    pct = avg_hr / (220 - age) if avg_hr else None
    intensity = 25 if pct is None else max(0, 40 * (1 - abs(pct - 0.75) / 0.5))  # best near 75% of est. max HR
    pace = min(30, avg_sp / 10 * 30)
    duration = min(20, dur / 45 * 20)
    cond = 10.0
    if temp and temp > 32:
        cond -= min(6, temp - 32)
    if hum and hum > 80:
        cond -= 3
    return int(round(max(0, min(100, intensity + pace + duration + max(0, cond)))))
 
 
def load_model():
    global MODEL, MODEL_NAME
    try:
        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        return
    if os.path.exists(MODEL_PATH):
        MODEL, MODEL_NAME = joblib.load(MODEL_PATH), "random-forest (trained, from file)"
        return
    rng = np.random.default_rng(42)
    X, y = [], []
    for _ in range(3000):
        hr, sp = rng.uniform(80, 185), rng.uniform(1, 14)
        f = [int(rng.integers(15, 60)), hr, hr + rng.uniform(5, 25), sp, sp + rng.uniform(0, 4), sp * rng.uniform(.1, 1.2),
             rng.uniform(5, 90), rng.uniform(50, 900), rng.uniform(15, 40), rng.uniform(30, 95)]
        X.append(f)
        y.append(category_idx(baseline_score(f)))
    MODEL = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)
    MODEL_NAME = "random-forest (baseline, synthetic training data)"
 
 
def analyze(summary, user, prev=None):
    t, h = summary.get("temperature") or {}, summary.get("humidity") or {}
    f = [user["age"], summary.get("avg_hr") or 0, summary.get("max_hr") or 0, summary.get("avg_speed") or 0,
         summary.get("max_speed") or 0, summary.get("distance_km") or 0, summary["duration_s"] / 60,
         summary.get("calories") or 0, t.get("avg") or 0, h.get("avg") or 0]
    score = baseline_score(f)
    cat = CATS[int(MODEL.predict([f])[0])] if MODEL else CATS[category_idx(score)]
    tips = []
    if prev and prev.get("summary"):
        ps, cs = prev["summary"].get("avg_speed") or 0, summary.get("avg_speed") or 0
        if ps and cs > ps * 1.03:
            tips.append("Your average speed improved compared with your previous session.")
        elif ps and cs < ps * 0.97:
            tips.append("Your average speed was lower than in your previous session.")
    if summary.get("avg_hr"):
        pct = summary["avg_hr"] / (220 - user["age"])
        tips.append("Heart-rate intensity was relatively high." if pct > 0.85 else
                    "Heart-rate intensity was in a steady aerobic range." if pct > 0.6 else
                    "Heart-rate intensity was low; you could push the pace if the goal is conditioning.")
    else:
        tips.append("No heart-rate data was recorded, so intensity could not be assessed.")
    if summary["duration_s"] >= 1800 and score >= 65:
        tips.append("Performance was strong for the session duration.")
    elif summary["duration_s"] < 600:
        tips.append("The session was short; longer sessions give more reliable analysis.")
    else:
        tips.append("Consider improving endurance with gradually longer sessions.")
    if (t.get("avg") or 0) > 32:
        tips.append("High temperature may have affected performance; hydrate well.")
    if not summary.get("distance_km"):
        tips.append("No distance was recorded (check the GPS signal).")
    return {"score": score, "category": cat, "model": MODEL_NAME, "insights": tips,
            "disclaimer": "Sports-performance analysis only. Not a medical diagnosis. Calories are estimates."}
 
 
# ----------------------------------------------------------------- live hub + session engine
class Hub:
    def __init__(self):
        self.rooms = {}
 
    async def add(self, sid, ws):
        self.rooms.setdefault(sid, set()).add(ws)
 
    def remove(self, sid, ws):
        self.rooms.get(sid, set()).discard(ws)
 
    async def send(self, sid, msg):
        for ws in list(self.rooms.get(sid, ())):
            try:
                await ws.send_json(msg)
            except Exception:
                self.remove(sid, ws)
 
 
hub, LIVE, DEMO_TASKS = Hub(), {}, {}
 
 
async def active_session(user, required=True):
    s = await db.sessions.find_one({"user_id": user["_id"], "status": {"$in": ["active", "paused"]}})
    if not s and required:
        raise HTTPException(409, "No active session. Start a session first.")
    return s
 
 
async def ingest(session, user, r, source):
    """Validate-already reading -> distance, calories, storage, WebSocket broadcast."""
    ts = now()
    lat, lon = r.get("latitude"), r.get("longitude")
    gps = lat is not None and lon is not None and not (lat == 0 and lon == 0)
    speed, dist, cal = r.get("speed"), session.get("distance_km", 0.0), session.get("calories", 0.0)
    last, upd, active = session.get("last_point"), {"last_reading_at": ts}, session["status"] == "active"
    if active:
        if gps:
            if last:
                d = haversine_km(last["lat"], last["lon"], lat, lon)
                if speed is None:
                    speed = d / max((ts - last["ts"]).total_seconds(), 0.001) * 3600 if d < 0.2 else 0
                if 0.002 <= d < 0.2:  # ignore GPS jitter (<2 m) and jumps (>200 m)
                    dist += d
                    upd["last_point"] = {"lat": lat, "lon": lon, "ts": ts}
            else:
                upd["last_point"] = {"lat": lat, "lon": lon, "ts": ts}
        prev = session.get("last_reading_at")
        if prev:  # kcal = MET x weight(kg) x hours (estimate)
            cal += met_for(speed or 0) * user["weight_kg"] * min((ts - prev).total_seconds(), 5) / 3600
    el = elapsed(session, ts)
    reading = {"session_id": session["_id"], "user_id": user["_id"], "device_id": r.get("device_id"), "source": source,
               "ts": ts, "elapsed_s": round(el, 1), "heart_rate": r.get("heart_rate"), "temperature": r.get("temperature"),
               "humidity": r.get("humidity"), "latitude": lat if gps else None, "longitude": lon if gps else None,
               "speed": round(speed, 2) if speed is not None else None, "distance_km": round(dist, 4), "calories": round(cal, 1)}
    if active:
        await db.sensor_readings.insert_one(dict(reading))
    upd.update(distance_km=dist, calories=cal)
    await db.sessions.update_one({"_id": session["_id"]}, {"$set": upd})
    payload = clean({**reading, "status": session["status"], "gps_fix": gps,
                     "avg_speed": round(dist / (el / 3600), 2) if el > 5 else 0})
    sid = str(session["_id"])
    LIVE[sid] = payload
    await hub.send(sid, payload)
    return payload
 
 
def summarize(s, readings):
    def col(k):
        return [r[k] for r in readings if r.get(k) is not None]
 
    def avg(a):
        return round(sum(a) / len(a), 1) if a else None
 
    def trio(a):
        return {"avg": avg(a), "min": round(min(a), 1), "max": round(max(a), 1)} if a else None
 
    dur, dist = elapsed(s), s.get("distance_km", 0.0)
    hr, sp = col("heart_rate"), col("speed")
    return {"duration_s": round(dur), "distance_km": round(dist, 3), "avg_speed": round(dist / (dur / 3600), 2) if dur > 5 else 0,
            "max_speed": round(max(sp), 1) if sp else 0, "avg_hr": avg(hr), "max_hr": round(max(hr)) if hr else None,
            "temperature": trio(col("temperature")), "humidity": trio(col("humidity")),
            "calories": round(s.get("calories", 0.0)), "readings": len(readings)}
 
 
async def demo_loop(sid, uid):
    lat, lon, heading, sp, hr = 30.7046, 76.7179, random.uniform(0, 360), 7.0, 95.0
    while True:
        await asyncio.sleep(1)
        try:
            s, u = await db.sessions.find_one({"_id": sid}), await db.users.find_one({"_id": uid})
            if not s or not u or s["status"] == "stopped":
                return
            sp = min(max(sp + random.uniform(-0.6, 0.6), 3), 14)
            heading += random.uniform(-12, 12)
            hr += (90 + sp * 7 - hr) * 0.05 + random.uniform(-1.5, 1.5)
            lat += math.cos(math.radians(heading)) * sp / 3600 / 111.0
            lon += math.sin(math.radians(heading)) * sp / 3600 / (111.0 * math.cos(math.radians(lat)))
            await ingest(s, u, {"device_id": "DEMO", "heart_rate": round(hr), "temperature": round(27 + random.uniform(-.4, .4), 1),
                                "humidity": round(62 + random.uniform(-1.5, 1.5), 1), "latitude": lat, "longitude": lon,
                                "speed": round(sp, 1)}, "demo")
        except PyMongoError:
            continue
 
 
# ----------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(_app):
    load_model()
    try:
        await db.users.create_index("email", unique=True)
        await db.devices.create_index("device_id", unique=True)
        await db.sessions.create_index([("user_id", 1), ("started_at", -1)])
        await db.sensor_readings.create_index([("session_id", 1), ("ts", 1)])
    except PyMongoError as e:
        print("WARNING: database unavailable at startup:", e)
    yield
    for t in DEMO_TASKS.values():
        t.cancel()
 
 
app = FastAPI(title="FieldSense Sports Performance Tracker API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
 
 
@app.exception_handler(PyMongoError)
async def db_down(_req: Request, _exc):
    return JSONResponse({"detail": "Database unavailable. Please try again shortly."}, status_code=503)
 
 
@app.get("/health")
async def health():
    try:
        await client.admin.command("ping")
        return {"status": "ok", "database": "up", "model": MODEL_NAME}
    except PyMongoError:
        return {"status": "degraded", "database": "down", "model": MODEL_NAME}
 
 
# ---- auth + profile
@app.post("/api/auth/signup", status_code=201)
async def signup(b: Signup):
    doc = b.model_dump()
    doc.update(email=b.email.lower(), password_hash=hash_pw(doc.pop("password")), created_at=now())
    if await db.users.find_one({"email": doc["email"]}):
        raise HTTPException(409, "An account with this email already exists.")
    doc["_id"] = (await db.users.insert_one(doc)).inserted_id
    return {"token": make_token(doc["_id"]), "user": clean(doc)}
 
 
@app.post("/api/auth/login")
async def login(b: Login):
    u = await db.users.find_one({"email": b.email.lower()})
    if not u or not check_pw(b.password, u["password_hash"]):
        raise HTTPException(401, "Incorrect email or password.")
    return {"token": make_token(u["_id"]), "user": clean(u)}
 
 
@app.get("/api/users/me")
async def me(user=Depends(current_user)):
    return clean(user)
 
 
@app.put("/api/users/me")
async def update_me(b: ProfileUpdate, user=Depends(current_user)):
    changes = b.model_dump(exclude_none=True)
    if changes:
        await db.users.update_one({"_id": user["_id"]}, {"$set": changes})
    return clean(await db.users.find_one({"_id": user["_id"]}))
 
 
# ---- sessions
@app.post("/api/session/start", status_code=201)
async def start_session(b: StartReq = StartReq(), user=Depends(current_user)):
    if await active_session(user, required=False):
        raise HTTPException(409, "A session is already running. Stop it before starting a new one.")
    s = {"user_id": user["_id"], "source": "demo" if b.demo else "device", "status": "active", "started_at": now(),
         "paused_total": 0.0, "distance_km": 0.0, "calories": 0.0}
    s["_id"] = (await db.sessions.insert_one(s)).inserted_id
    if b.demo:
        DEMO_TASKS[s["_id"]] = asyncio.create_task(demo_loop(s["_id"], user["_id"]))
    return clean(s)
 
 
@app.post("/api/session/pause")
async def pause_session(user=Depends(current_user)):
    s = await active_session(user)
    if s["status"] != "active":
        raise HTTPException(409, "Session is already paused.")
    await db.sessions.update_one({"_id": s["_id"]}, {"$set": {"status": "paused", "paused_at": now()}})
    return clean(await db.sessions.find_one({"_id": s["_id"]}))
 
 
@app.post("/api/session/resume")
async def resume_session(user=Depends(current_user)):
    s = await active_session(user)
    if s["status"] != "paused":
        raise HTTPException(409, "Session is not paused.")
    extra = (now() - s["paused_at"]).total_seconds()
    await db.sessions.update_one({"_id": s["_id"]}, {"$set": {"status": "active", "last_point": None},
                                                      "$inc": {"paused_total": extra}, "$unset": {"paused_at": ""}})
    return clean(await db.sessions.find_one({"_id": s["_id"]}))
 
 
@app.post("/api/session/stop")
async def stop_session(user=Depends(current_user)):
    s = await active_session(user)
    ts = now()
    if s["status"] == "paused" and s.get("paused_at"):
        s["paused_total"] = s.get("paused_total", 0) + (ts - s["paused_at"]).total_seconds()
    s.update(status="stopped", ended_at=ts)
    readings = await db.sensor_readings.find({"session_id": s["_id"]}).sort("ts", 1).to_list(50000)
    summary = summarize(s, readings)
    prev = await db.sessions.find_one({"user_id": user["_id"], "status": "stopped", "source": s["source"]}, sort=[("ended_at", -1)])
    ai = analyze(summary, user, prev)
    await db.sessions.update_one({"_id": s["_id"]}, {"$set": {"status": "stopped", "ended_at": ts, "paused_total": s["paused_total"],
                                                              "summary": summary, "ai": ai}, "$unset": {"paused_at": "", "last_point": ""}})
    task = DEMO_TASKS.pop(s["_id"], None)
    if task:
        task.cancel()
    LIVE.pop(str(s["_id"]), None)
    return clean(await db.sessions.find_one({"_id": s["_id"]}))
 
 
def source_filter(source):
    return {} if source == "all" else {"source": "demo" if source == "demo" else "device"}
 
 
@app.get("/api/session/history")
async def history(source: str = "device", limit: int = 100, user=Depends(current_user)):
    q = {"user_id": user["_id"], "status": "stopped", **source_filter(source)}
    return clean(await db.sessions.find(q).sort("started_at", -1).to_list(min(limit, 500)))
 
 
@app.get("/api/session/active")
async def active(user=Depends(current_user)):
    s = await active_session(user, required=False)
    return {"session": clean(s) if s else None, "live": LIVE.get(str(s["_id"])) if s else None}
 
 
@app.get("/api/session/{session_id}")
async def session_detail(session_id: str, user=Depends(current_user)):
    s = await db.sessions.find_one({"_id": oid(session_id), "user_id": user["_id"]})
    if not s:
        raise HTTPException(404, "Session not found")
    rs = await db.sensor_readings.find({"session_id": s["_id"]}).sort("ts", 1).to_list(50000)
    step = max(1, math.ceil(len(rs) / 600))
    out = clean(s)
    out["readings"] = clean(rs[::step])
    return out
 
 
@app.get("/api/session/{session_id}/live")
async def session_live(session_id: str, user=Depends(current_user)):
    if not await db.sessions.find_one({"_id": oid(session_id), "user_id": user["_id"]}):
        raise HTTPException(404, "Session not found")
    return LIVE.get(session_id, {})
 
 
# ---- device
@app.post("/api/device/register")
async def register_device(b: DeviceReg, user=Depends(current_user)):
    d = await db.devices.find_one({"device_id": b.device_id})
    if d and d["user_id"] != user["_id"]:
        raise HTTPException(409, "This device ID belongs to another account.")
    await db.devices.update_one({"device_id": b.device_id}, {"$setOnInsert": {"user_id": user["_id"], "created_at": now()}}, upsert=True)
    return {"device_id": b.device_id, "registered": True}
 
 
@app.post("/api/device/data")
async def device_data(b: DeviceReading, x_device_key: Optional[str] = Header(None)):
    if DEVICE_KEY and x_device_key != DEVICE_KEY:
        raise HTTPException(401, "Invalid device key")
    dev = await db.devices.find_one({"device_id": b.device_id})
    if not dev:
        raise HTTPException(404, "Device not registered. Register it in the app first.")
    data = b.model_dump()
    await db.devices.update_one({"_id": dev["_id"]}, {"$set": {"last_seen": now(), "last_reading": data, "gps_fix": data["latitude"] is not None}})
    user = await db.users.find_one({"_id": dev["user_id"]})
    s = await db.sessions.find_one({"user_id": dev["user_id"], "source": "device", "status": {"$in": ["active", "paused"]}})
    if not s or not user:
        return {"stored": False, "detail": "No active session; reading not stored."}
    p = await ingest(s, user, data, "device")
    return {"stored": s["status"] == "active", "session_id": str(s["_id"]), "distance_km": p["distance_km"]}
 
 
@app.get("/api/device/status")
async def device_status(user=Depends(current_user)):
    out = []
    for d in await db.devices.find({"user_id": user["_id"]}).to_list(20):
        seen = d.get("last_seen")
        out.append({"device_id": d["device_id"], "last_seen": seen.isoformat() if seen else None, "last_reading": d.get("last_reading"),
                    "connected": bool(seen and (now() - seen).total_seconds() < DEVICE_TIMEOUT_S), "gps_fix": d.get("gps_fix", False)})
    return {"devices": out, "connected": any(x["connected"] for x in out)}
 
 
# ---- analytics + AI
PERIODS = {"7d": 7, "30d": 30, "3m": 90, "all": None}
 
 
async def build_analytics(user, period, source):
    if period not in PERIODS:
        raise HTTPException(404, "Unknown period. Use 7d, 30d, 3m or all.")
    q = {"user_id": user["_id"], "status": "stopped", **source_filter(source)}
    allx = await db.sessions.find(q).sort("started_at", 1).to_list(2000)
    t, days = now(), PERIODS[period]
    sel = [s for s in allx if not days or s["started_at"] >= t - timedelta(days=days)]
 
    def dist_last(n):
        return round(sum(s["summary"]["distance_km"] for s in allx if s["started_at"] >= t - timedelta(days=n)), 2)
 
    daily = {}
    for s in sel:
        d = daily.setdefault(s["started_at"].date().isoformat(), {"distance_km": 0, "calories": 0, "sessions": 0, "scores": []})
        d["distance_km"] += s["summary"]["distance_km"]; d["calories"] += s["summary"]["calories"]; d["sessions"] += 1
        d["scores"].append(s["ai"]["score"])
    series = [{"date": k, "distance_km": round(v["distance_km"], 2), "calories": v["calories"], "sessions": v["sessions"],
               "score": round(sum(v["scores"]) / len(v["scores"]))} for k, v in sorted(daily.items())]
    hrs = [s["summary"]["avg_hr"] for s in sel if s["summary"].get("avg_hr")]
    maxes = [s["summary"]["max_hr"] for s in sel if s["summary"].get("max_hr")]
    dist, secs = sum(s["summary"]["distance_km"] for s in sel), sum(s["summary"]["duration_s"] for s in sel)
    return {"period": period, "source": source, "weekly_distance_km": dist_last(7), "monthly_distance_km": dist_last(30), "daily": series,
            "totals": {"sessions": len(sel), "distance_km": round(dist, 2), "calories": sum(s["summary"]["calories"] for s in sel),
                       "active_time_s": secs, "avg_hr": round(sum(hrs) / len(hrs), 1) if hrs else None,
                       "max_hr": max(maxes) if maxes else None, "avg_speed": round(dist / (secs / 3600), 2) if secs else 0}}
 
 
@app.get("/api/analytics")
async def analytics(source: str = "device", user=Depends(current_user)):
    return await build_analytics(user, "30d", source)
 
 
@app.get("/api/analytics/{period}")
async def analytics_period(period: str, source: str = "device", user=Depends(current_user)):
    return await build_analytics(user, period, source)
 
 
@app.get("/api/ai/performance/{session_id}")
async def ai_performance(session_id: str, user=Depends(current_user)):
    s = await db.sessions.find_one({"_id": oid(session_id), "user_id": user["_id"]})
    if not s:
        raise HTTPException(404, "Session not found")
    if s["status"] != "stopped":
        raise HTTPException(409, "Stop the session to generate its analysis.")
    return clean(s.get("ai") or analyze(s["summary"], user))
 
 
# ---- demo
@app.get("/api/demo/data")
async def demo_data():
    """One simulated reading for previews. Never stored. Use POST /api/session/start {"demo": true} for a full demo session."""
    return {"demo": True, "device_id": "DEMO", "heart_rate": random.randint(110, 150), "temperature": round(random.uniform(26, 29), 1),
            "humidity": round(random.uniform(58, 66), 1), "latitude": 30.7046 + random.uniform(-.001, .001),
            "longitude": 76.7179 + random.uniform(-.001, .001), "speed": round(random.uniform(5, 10), 1)}
 
 
# ---- websocket
@app.websocket("/ws/live/{session_id}")
async def ws_live(ws: WebSocket, session_id: str, token: str = ""):
    try:
        uid = decode_token(token)
        ok = await db.sessions.find_one({"_id": ObjectId(session_id), "user_id": ObjectId(uid)})
    except Exception:
        ok = None
    if not ok:
        await ws.close(code=4401)
        return
    await ws.accept()
    await hub.add(session_id, ws)
    if session_id in LIVE:
        await ws.send_json(LIVE[session_id])
    try:
        while True:
            await ws.receive_text()  # keep-alive pings from the browser
    except WebSocketDisconnect:
        pass
    finally:
        hub.remove(session_id, ws)
 
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
 
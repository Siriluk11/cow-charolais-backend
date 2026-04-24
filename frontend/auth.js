const AUTH_USERS_KEY = "charolais_users_v3";
const AUTH_SESSION_KEY = "charolais_session_v3";
const PROFILE_KEY_PREFIX = "charolais_profile_";

function normalizeUsername(username) {
    return String(username || "").trim();
}

function isValidUsername(username) {
    const value = normalizeUsername(username);
    return /^[A-Za-zก-๙0-9_.-]{2,30}$/u.test(value);
}

function isValidPassword(password) {
    return typeof password === "string" && password.length >= 8 && /[A-Z]/.test(password);
}

async function sha256(text) {
    const data = new TextEncoder().encode(text);
    const hashBuffer = await crypto.subtle.digest("SHA-256", data);
    return Array.from(new Uint8Array(hashBuffer))
        .map(byte => byte.toString(16).padStart(2, "0"))
        .join("");
}

function getUsers() {
    try {
        const raw = localStorage.getItem(AUTH_USERS_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch {
        return [];
    }
}

function saveUsers(users) {
    localStorage.setItem(AUTH_USERS_KEY, JSON.stringify(users));
}

async function ensureDefaultAdmin() {
    const users = getUsers();
    const hasAdmin = users.some(user => user.username.toLowerCase() === "admin");

    if (!hasAdmin) {
        users.push({
            username: "admin",
            passwordHash: await sha256("Admin1234"),
            role: "admin",
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString()
        });
        saveUsers(users);
    }
}

function getCurrentUser() {
    try {
        const raw = localStorage.getItem(AUTH_SESSION_KEY);
        return raw ? JSON.parse(raw) : null;
    } catch {
        return null;
    }
}

function requireLogin() {
    const session = getCurrentUser();
    if (!session || !session.username) {
        window.location.href = "index.html";
    }
}

function requireAdmin() {
    const session = getCurrentUser();
    if (!session || session.role !== "admin") {
        window.location.href = "index.html";
    }
}

function logout() {
    localStorage.removeItem(AUTH_SESSION_KEY);
    window.location.href = "index.html";
}

async function registerUser(username, password) {
    await ensureDefaultAdmin();

    username = normalizeUsername(username);

    if (!isValidUsername(username)) {
        return {
            ok: false,
            message: "ชื่อผู้ใช้ต้องมี 2–30 ตัว ใช้ภาษาไทย อังกฤษ ตัวเลข จุด ขีดกลาง หรือขีดล่างได้"
        };
    }

    if (username.toLowerCase() === "admin") {
        return {
            ok: false,
            message: "ไม่สามารถใช้ชื่อผู้ใช้ admin ได้"
        };
    }

    if (!isValidPassword(password)) {
        return {
            ok: false,
            message: "รหัสผ่านต้องมีอย่างน้อย 8 ตัว และมีตัวอักษรภาษาอังกฤษตัวใหญ่ 1 ตัว"
        };
    }

    const users = getUsers();
    const duplicated = users.some(user => user.username.toLowerCase() === username.toLowerCase());

    if (duplicated) {
        return {
            ok: false,
            message: "ชื่อผู้ใช้นี้ถูกใช้แล้ว"
        };
    }

    const passwordHash = await sha256(password);

    users.push({
        username,
        passwordHash,
        role: "user",
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
    });

    saveUsers(users);

    localStorage.setItem(PROFILE_KEY_PREFIX + username, JSON.stringify({
        username,
        farmName: "",
        avatar: "",
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
    }));

    return {
        ok: true,
        message: "ลงทะเบียนสำเร็จ สามารถเข้าสู่ระบบได้แล้ว"
    };
}

async function loginUser(username, password) {
    await ensureDefaultAdmin();

    username = normalizeUsername(username);

    if (!username || !password) {
        return {
            ok: false,
            message: "กรุณากรอกชื่อผู้ใช้และรหัสผ่าน"
        };
    }

    const users = getUsers();
    const user = users.find(item => item.username.toLowerCase() === username.toLowerCase());

    if (!user) {
        return {
            ok: false,
            message: "ไม่พบบัญชีผู้ใช้นี้"
        };
    }

    const passwordHash = await sha256(password);

    if (passwordHash !== user.passwordHash) {
        return {
            ok: false,
            message: "รหัสผ่านไม่ถูกต้อง"
        };
    }

    const role = user.role || "user";

    localStorage.setItem(AUTH_SESSION_KEY, JSON.stringify({
        username: user.username,
        role,
        loginAt: new Date().toISOString()
    }));

    return {
        ok: true,
        role,
        message: "เข้าสู่ระบบสำเร็จ"
    };
}

async function resetPassword(username, newPassword) {
    await ensureDefaultAdmin();

    username = normalizeUsername(username);

    if (!username) {
        return {
            ok: false,
            message: "กรุณากรอกชื่อผู้ใช้"
        };
    }

    if (!isValidPassword(newPassword)) {
        return {
            ok: false,
            message: "รหัสผ่านใหม่ต้องมีอย่างน้อย 8 ตัว และมีตัวอักษรภาษาอังกฤษตัวใหญ่ 1 ตัว"
        };
    }

    const users = getUsers();
    const index = users.findIndex(user => user.username.toLowerCase() === username.toLowerCase());

    if (index === -1) {
        return {
            ok: false,
            message: "ไม่พบบัญชีผู้ใช้นี้"
        };
    }

    users[index].passwordHash = await sha256(newPassword);
    users[index].updatedAt = new Date().toISOString();

    saveUsers(users);
    localStorage.removeItem(AUTH_SESSION_KEY);

    return {
        ok: true,
        message: "เปลี่ยนรหัสผ่านสำเร็จ กรุณาเข้าสู่ระบบใหม่"
    };
}

function getProfile() {
    const session = getCurrentUser();
    if (!session || !session.username) return null;

    try {
        const raw = localStorage.getItem(PROFILE_KEY_PREFIX + session.username);
        return raw ? JSON.parse(raw) : {
            username: session.username,
            farmName: "",
            avatar: ""
        };
    } catch {
        return {
            username: session.username,
            farmName: "",
            avatar: ""
        };
    }
}

function saveProfile(profile) {
    const session = getCurrentUser();
    if (!session || !session.username) return false;

    const oldProfile = getProfile() || {};

    const payload = {
        username: session.username,
        farmName: profile.farmName || "",
        avatar: profile.avatar || oldProfile.avatar || "",
        updatedAt: new Date().toISOString()
    };

    localStorage.setItem(PROFILE_KEY_PREFIX + session.username, JSON.stringify(payload));
    return true;
}
function saveActivityLog(action, detail = "") {
    const key = "charolais_activity_logs";
    let logs = [];

    try {
        logs = JSON.parse(localStorage.getItem(key)) || [];
    } catch {
        logs = [];
    }

    const session = getCurrentUser();

    logs.unshift({
        username: session?.username || "guest",
        role: session?.role || "unknown",
        action,
        detail,
        createdAt: new Date().toLocaleString("th-TH")
    });

    localStorage.setItem(key, JSON.stringify(logs.slice(0, 200)));
}
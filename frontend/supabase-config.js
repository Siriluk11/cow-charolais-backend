// 🔗 URL ของ Render backend (ของคุณ)
const API_URL = "https://cow-charolais-backend.onrender.com";

// =========================
// SCAN RESULTS (SAVE)
// =========================
async function saveScanResultToDB({ className, score, confidence, sourceType }) {
    try {
        const payload = {
            class_name: className,
            score: score,
            confidence: confidence,
            source_type: sourceType,
            image_url: null
        };

        const response = await fetch(`${API_URL}/save-result`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const text = await response.text();
            console.error("❌ saveScanResultToDB:", text);
            return null;
        }

        const data = await response.json();
        console.log("✅ saved:", data);
        return data;

    } catch (err) {
        console.error("❌ saveScanResultToDB error:", err);
        return null;
    }
}


// =========================
// HISTORY
// =========================
async function getScanHistoryFromDB(limit = 5) {
    try {
        const response = await fetch(`${API_URL}/history?limit=${limit}`);

        if (!response.ok) {
            const text = await response.text();
            console.error("❌ history error:", text);
            return [];
        }

        const data = await response.json();
        console.log("✅ history:", data);
        return Array.isArray(data) ? data : [];

    } catch (err) {
        console.error("❌ history fetch error:", err);
        return [];
    }
}


// =========================
// PROFILE (ถ้ามีใช้)
// =========================
async function getProfileFromDB() {
    try {
        const response = await fetch(`${API_URL}/profile`);
        if (!response.ok) return null;
        return await response.json();
    } catch (err) {
        console.error("❌ getProfile error:", err);
        return null;
    }
}

async function saveProfileToDB({ username, farmName, avatarUrl }) {
    try {
        const response = await fetch(`${API_URL}/profile`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                username: username || "",
                farm_name: farmName || "",
                avatar_url: avatarUrl || ""
            })
        });

        if (!response.ok) {
            const text = await response.text();
            console.error("❌ saveProfile error:", text);
            return null;
        }

        return await response.json();

    } catch (err) {
        console.error("❌ saveProfile error:", err);
        return null;
    }
}


// =========================
// STORAGE (ถ้ามีใช้)
// =========================
async function uploadProfileImage(file) {
    try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(`${API_URL}/upload-profile-image`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const text = await response.text();
            console.error("❌ upload image error:", text);
            return null;
        }

        const data = await response.json();
        return data.public_url || data.url || null;

    } catch (err) {
        console.error("❌ upload image error:", err);
        return null;
    }
}
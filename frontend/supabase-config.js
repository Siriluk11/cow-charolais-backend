const API_BASE_URL = "https://cow-charolais-backend.onrender.com";

// =========================
// SCAN RESULTS
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

        const response = await fetch(`${API_BASE_URL}/save-result`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const text = await response.text();
            alert("บันทึกไม่สำเร็จ: " + text);
            return null;
        }

        return await response.json();
    } catch (err) {
        console.error("saveScanResultToDB:", err);
        alert("บันทึกไม่สำเร็จ: " + err.message);
        return null;
    }
}

async function getScanHistoryFromDB(limit = 5) {
    try {
        const response = await fetch(`${API_BASE_URL}/history?limit=${limit}`);

        if (!response.ok) {
            const text = await response.text();
            alert("โหลดประวัติไม่สำเร็จ: " + text);
            return [];
        }

        return await response.json();
    } catch (err) {
        console.error("getScanHistoryFromDB:", err);
        alert("โหลดประวัติไม่สำเร็จ: " + err.message);
        return [];
    }
}

// =========================
// PROFILE
// =========================
async function getProfileFromDB() {
    try {
        const response = await fetch(`${API_BASE_URL}/profile`);
        if (!response.ok) return null;
        return await response.json();
    } catch (err) {
        console.error("getProfileFromDB:", err);
        return null;
    }
}

async function saveProfileToDB({ username, farmName, avatarUrl }) {
    try {
        const response = await fetch(`${API_BASE_URL}/profile`, {
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
            alert("บันทึกโปรไฟล์ไม่สำเร็จ: " + text);
            return null;
        }

        return await response.json();
    } catch (err) {
        console.error("saveProfileToDB:", err);
        alert("บันทึกโปรไฟล์ไม่สำเร็จ: " + err.message);
        return null;
    }
}

async function uploadProfileImage(file) {
    try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(`${API_BASE_URL}/upload-profile-image`, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const text = await response.text();
            alert("อัปโหลดรูปไม่สำเร็จ: " + text);
            return null;
        }

        const data = await response.json();
        return data.public_url || data.url || null;
    } catch (err) {
        console.error("uploadProfileImage:", err);
        alert("อัปโหลดรูปไม่สำเร็จ: " + err.message);
        return null;
    }
}
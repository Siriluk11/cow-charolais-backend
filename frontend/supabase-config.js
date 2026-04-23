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
            console.error("saveScanResultToDB error:", text);
            return null;
        }

        return await response.json();
    } catch (err) {
        console.error("saveScanResultToDB error:", err);
        return null;
    }
}

async function getScanHistoryFromDB(limit = 5) {
    try {
        const response = await fetch(`${API_URL}/history?limit=${limit}`);

        if (!response.ok) {
            const text = await response.text();
            console.error("getScanHistoryFromDB error:", text);
            return [];
        }

        const data = await response.json();
        return Array.isArray(data) ? data : [];
    } catch (err) {
        console.error("getScanHistoryFromDB error:", err);
        return [];
    }
}

async function getProfileFromDB() {
    try {
        const response = await fetch(`${API_URL}/profile`);

        if (!response.ok) {
            return null;
        }

        return await response.json();
    } catch (err) {
        console.error("getProfileFromDB error:", err);
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
            console.error("saveProfileToDB error:", text);
            return null;
        }

        return await response.json();
    } catch (err) {
        console.error("saveProfileToDB error:", err);
        return null;
    }
}

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
            console.error("uploadProfileImage error:", text);
            return null;
        }

        const data = await response.json();
        return data.public_url || data.url || null;
    } catch (err) {
        console.error("uploadProfileImage error:", err);
        return null;
    }
}
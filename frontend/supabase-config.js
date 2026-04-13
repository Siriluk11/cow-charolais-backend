const SUPABASE_URL = "https://momddwmijtaszufzrhll.supabase.co";
const SUPABASE_ANON_KEY = "sb_publishable_3elCSxWZHTYTZAPQyeIx6w_jbjnit8q";

const sb = supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// =========================
// SCAN RESULTS
// =========================
async function saveScanResultToDB({ className, score, confidence, sourceType }) {
    const payload = {
        class_name: className,
        score: score,
        confidence: confidence,
        source_type: sourceType,
        image_url: null
    };

    const { data, error } = await sb
        .from("scan_results")
        .insert([payload])
        .select();

    if (error) {
        console.error("saveScanResultToDB error:", error);
        throw error;
    }

    return data;
}

async function getScanHistoryFromDB(limit = 5) {
    const { data, error } = await sb
        .from("scan_results")
        .select("*")
        .order("created_at", { ascending: false })
        .limit(limit);

    if (error) {
        console.error("getScanHistoryFromDB error:", error);
        throw error;
    }

    return data || [];
}

// =========================
// PROFILE
// =========================
async function getProfileFromDB() {
    const { data, error } = await sb
        .from("profiles")
        .select("*")
        .order("created_at", { ascending: false })
        .limit(1)
        .maybeSingle();

    if (error) {
        console.error("getProfileFromDB error:", error);
        throw error;
    }

    return data || null;
}

async function saveProfileToDB({ username, farmName, avatarUrl }) {
    const existing = await getProfileFromDB();

    if (existing?.id) {
        const { data, error } = await sb
            .from("profiles")
            .update({
                username: username || "",
                farm_name: farmName || "",
                avatar_url: avatarUrl || ""
            })
            .eq("id", existing.id)
            .select()
            .single();

        if (error) {
            console.error("saveProfileToDB update error:", error);
            throw error;
        }

        return data;
    } else {
        const { data, error } = await sb
            .from("profiles")
            .insert([{
                username: username || "",
                farm_name: farmName || "",
                avatar_url: avatarUrl || ""
            }])
            .select()
            .single();

        if (error) {
            console.error("saveProfileToDB insert error:", error);
            throw error;
        }

        return data;
    }
}

// =========================
// STORAGE
// =========================
async function uploadProfileImage(file) {
    const ext = file.name.split(".").pop() || "png";
    const fileName = `profile_${Date.now()}.${ext}`;

    const { error } = await sb.storage
        .from("profile-images")
        .upload(fileName, file, {
            upsert: true
        });

    if (error) {
        console.error("uploadProfileImage error:", error);
        throw error;
    }

    const { data } = sb.storage
        .from("profile-images")
        .getPublicUrl(fileName);

    return data.publicUrl;
}
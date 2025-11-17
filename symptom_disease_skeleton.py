
import sys
from typing import List, Dict, Set, Tuple, Optional, Iterable
from collections import defaultdict
import random

SUPPORTED_LANGUAGES: Tuple[str, ...] = ("en", "hi", "gu", "te")
LANGUAGE_FALLBACKS: Dict[str, Tuple[str, ...]] = {
    "en": ("en", "hi", "gu", "te"),
    "hi": ("hi", "en", "gu", "te"),
    "gu": ("gu", "hi", "en", "te"),
    "te": ("te", "hi", "en", "gu"),
}


def _coerce_question_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _select_text_candidate(texts: Dict[str, str], candidates: Tuple[str, ...]) -> str:
    for key in candidates:
        raw_value = texts.get(key)
        if raw_value is None:
            continue
        value = _coerce_question_text(raw_value).strip()
        if value:
            return value
    return ""


def _normalize_question_texts(texts: Optional[Dict[str, str]]) -> Dict[str, str]:
    if not isinstance(texts, dict):
        base_text = _coerce_question_text(texts)
        return {lang: base_text for lang in SUPPORTED_LANGUAGES}

    normalized: Dict[str, str] = {}
    for lang in SUPPORTED_LANGUAGES:
        normalized[lang] = _select_text_candidate(texts, LANGUAGE_FALLBACKS[lang])

    fallback = next((text for text in normalized.values() if text), "")
    if fallback:
        for lang, text in normalized.items():
            if not text:
                normalized[lang] = fallback

    return normalized

def _normalize_category_value(category: Optional[str]) -> Optional[str]:
    """Return a normalized category string for comparison."""

    if category is None:
        return None
    if not isinstance(category, str):
        category = str(category)
    normalized = category.strip().lower()
    return normalized or None


def _category_prefix(normalized_category: Optional[str]) -> Optional[str]:
    if not normalized_category:
        return None
    if ":" in normalized_category:
        return normalized_category.split(":", 1)[0].strip()
    return normalized_category


def categories_conflict(category: Optional[str], seen_categories: Iterable[Optional[str]]) -> bool:
    normalized = _normalize_category_value(category)
    if not normalized:
        return False

    prefix = _category_prefix(normalized)

    for seen in seen_categories:
        seen_normalized = _normalize_category_value(seen)
        if not seen_normalized:
            continue

        if normalized == seen_normalized:
            return True

        seen_prefix = _category_prefix(seen_normalized)
        if prefix and seen_prefix and prefix == seen_prefix:
            return True

        if normalized in seen_normalized or seen_normalized in normalized:
            return True

    return False

def make_question(texts: Dict[str, str], weight: float, q_type: str,
                  category: Optional[str] = None,
                  symptom: Optional[str] = None,
                  risk_factor: bool = False) -> Tuple[Dict[str, str], float, str]:
    """Build a standard question tuple with optional metadata."""
    q_dict: Dict[str, str] = {}
    if isinstance(texts, dict):
        q_dict.update(texts)
    else:
        q_dict["en"] = _coerce_question_text(texts)

    q_dict.update(_normalize_question_texts(texts))
    q_dict["category"] = category
    q_dict["symptom"] = symptom
    q_dict["risk_factor"] = risk_factor
    return (q_dict, weight, q_type)

class SymptomNode:
    def __init__(self, name, clarifying_questions=None, followup_branches=None, weight=1.0):
        self.name = name
        self.clarifying_questions = clarifying_questions or []
        self.followup_branches = followup_branches or []
        self.weight = weight  # Importance weight for this symptom
        self.visited = False
        self.children = []
        self.confidence_score = 0.0

    def add_child(self, child):
        self.children.append(child)

class HybridSymptomNode:
    def __init__(self, name, component_symptoms, hybrid_questions=None, weight=1.0,
                 question_sets=None):
        self.name = name
        self.component_symptoms = set(component_symptoms)  # e.g., {"Fever", "Cough"}
        self.component_symptoms_normalized = {
            symptom.lower() for symptom in self.component_symptoms
        }
        self.hybrid_questions = hybrid_questions or []
        self.weight = weight
        self.visited = False
        self.children = []
        self.confidence_score = 0.0
        self.question_sets = question_sets or {}

    def add_child(self, child):
        self.children.append(child)

class DiseaseNode:
    def __init__(self, name, required_symptoms, diagnostic_questions=None, symptom_weights=None):
        self.name = name
        self.required_symptoms = set(required_symptoms)
        self.diagnostic_questions = diagnostic_questions or []
        # Weights for how important each symptom is for this disease
        self.symptom_weights = symptom_weights or {s: 1.0 for s in required_symptoms}
        self.confidence_threshold = 0.6  # Minimum confidence to consider this disease

class DiagnosticEngine:

    def __init__(self):
        self.symptom_tree_roots = {}
        self.diseases = []
        self.confirmed_symptoms = set()
        self.symptom_confidences = {}
        self.disease_scores = {}
        self.symptom_details = {}
        self.hybrid_symptom_nodes = {}
        self.triggered_hybrids = set()
        self.gi_hybrid_key = "stomach_pain_acidity_gas_bloating"
        self.gi_question_lookup = {}
        self.gi_question_sets = self.build_gi_question_sets()
        self.gi_followup_responses = {}
        self.asked_gi_sets = set()

    def build_symptom_tree(self):
        """Build comprehensive symptom trees with bilingual questions (EN + HI for all questions)"""
        """
        # Enhanced Fever Tree
        fever = SymptomNode("fever", [
            #({"en": "For how many days has the fever been present?", "hi": "बुखार आपको कितने दिनों से है?"}, 0.9, "number"),
            make_question(
                {"en": "Did the fever start suddenly, or did it develop gradually?",
                "hi": "क्या बुखार अचानक शुरू हुआ, या धीरे-धीरे बढ़ा?"},
                0.8, "choice",
                category="type: fever",
                symptom="fever"
            ),
           
            make_question(
                {"en": "Do you experience shivering along with the fever?",
                "hi": "क्या बुखार के साथ आपको कंपकंपी होती है?"},
                0.9, "yesno",
                category="shivering",
                symptom="shivering"
            ),
            make_question(
                {"en": "Does the temperature tend to rise in the evening?",
                "hi": "क्या शाम के समय बुखार बढ़ जाता है?"},
                0.7, "yesno",
                category="instance: fevr",
                symptom=None
            ),

            make_question(
                {"en": "Are you experiencing headaches?",
                "hi": "क्या आपको सिरदर्द है?"},
                0.7, "yesno",
                category="headache",
                symptom="headache"
            ),

            
        ], weight=1.0)

        fever_chills = SymptomNode("chills", [
            make_question(
                {"en": "Do you experience shivering along with the fever?",
                "hi": "क्या बुखार के साथ आपको कंपकंपी होती है?"},
                0.9, "yesno",
                category="shivering",
                symptom="shivering"
            ),
            make_question(
                {"en": "For how many minutes does the shivering usually last?",
                "hi": "कंपकंपी आमतौर पर कितने मिनट तक रहती है?"},
                0.8, "number",
                category="duration: chills",
                symptom="chills"
            ),
            make_question(
                {"en": "Do you sweat after the shivering stops?",
                "hi": "क्या कंपकंपी के बाद आपको पसीना आता है?"},
                0.8, "yesno",
                category="sweat",
                symptom="sweating"
            ),
        ])
        fever_headache = SymptomNode("headache", [
            make_question(
                {"en": "Is the headache steady or throbbing in nature?",
                "hi": "क्या सिरदर्द लगातार होता है या धड़कन जैसा होता है?"},
                0.8, "choice",
                category="type: headache",
                symptom=None
            ),
            make_question(
                {"en": "Where is the headache located: front or back of the head?",
                "hi": "सिरदर्द कहाँ होता है: माथे पर या सिर के पीछे?"},
                0.7, "choice",
                category="location: headache",
                symptom=None
            ),
            make_question(
                {"en": "Do you have pain behind the eyes?",
                "hi": "क्या आँखों के पीछे दर्द होता है?"},
                0.8, "yesno",
                category="pain: eyes",
                symptom=None
            ),
        ])
        fever_night_sweats = SymptomNode("night sweats", [
            make_question(
            {"en": "Do you wake up at night drenched in sweat?",
            "hi": "क्या आप रात में पसीने से भीगे हुए उठते हैं?"},
            0.9, "yesno",
            category="sweating",
            symptom="diet:sweating"
        ),
            
        ])

        fever.add_child(fever_chills)
        fever.add_child(fever_headache)
        fever.add_child(fever_night_sweats)
        self.symptom_tree_roots["fever"] = fever
        
        # Enhanced Cough Tree
        cough = SymptomNode("cough", [
            make_question(
                {"en": "For how many days have you had the cough?",
                "hi": "आपको खांसी कितने दिनों से है?"},
                0.9, "number",
                category="duration: cough",
                symptom=None
            ),
            make_question(
                {"en": "At the beginning, was the cough severe, or was it mild and getting worse?",
                "hi": "शुरुआत में खांसी तेज़ थी, या हल्की थी और धीरे-धीरे बढ़ रही है?"},
                0.8, "choice",
                category="instance: cough",
                symptom=None
            ),
            make_question(
                {"en": "Do you bring up phlegm (sputum) when you cough?",
                "hi": "क्या खांसी के साथ बलगम निकलता है?"},
                0.9, "choice",
                category="phlegm",
                symptom="phlegm"
            ),
            make_question(
                {"en": "Do smoke, dust, or cold air make the cough worse?",
                "hi": "क्या धुआँ, धूल या ठंडी हवा से खांसी बढ़ती है?"},
                0.7, "yesno",
                category="environmental_triggers",
                symptom="cough"
            ),
            make_question(
                {"en": "Do you hear wheezing or a whistling sound when breathing?",
                "hi": "क्या सांस लेते समय घरघराहट या सीटी जैसी आवाज़ आती है?"},
                0.7, "yesno",
                category="wheezing",
                symptom="wheezing"
            ),
        ])

        productive_cough = SymptomNode("productive cough", [
            make_question(
                {"en": "What is the color of the phlegm: clear, white, yellow, green, or rusty?",
                "hi": "बलगम का रंग क्या है: साफ, सफेद, पीला, हरा, या जंग जैसा?"},
                0.9, "choice",
                category="phlegm color",
                symptom="phlegm"
            ),
            make_question(
                {"en": "What is the consistency of the phlegm: mucoid (sticky) or purulent (pus-like)?",
                "hi": "बलगम की प्रकृति कैसी है: चिपचिपा (म्यूकॉयड) या मवाद जैसा (प्यूरुलेंट)?"},
                0.9, "choice",
                category="sputum_consistency",
                symptom="phlegm"
            ),
            make_question(
                {"en": "Do you notice any blood in the phlegm?",
                "hi": "क्या बलगम में खून दिखाई देता है?"},
                0.9, "yesno",
                category="blood: phlem",
                symptom="hemoptysis"
            ),
            
        ])

        dry_cough = SymptomNode("dry cough", [
            make_question(
                {"en": "Does the cough cause throat irritation?",
                "hi": "क्या खांसी से गले में जलन या खराश होती है?"},
                0.7, "yesno",
                category="throat_irritation",
                symptom="dry_cough"
            ),
            make_question(
                {"en": "Do triggers like dust or smoke make the cough worse?",
                "hi": "क्या धूल या धुआँ जैसे कारणों से खांसी बढ़ती है?"},
                0.7, "yesno",
                category="cause: cough",
                symptom="dry_cough"
            ),
            
        ])

        hemoptysis = SymptomNode("hemoptysis", [
             make_question(
                {"en": "Have you noticed any blood when you cough?",
                "hi": "क्या खांसते समय खून आता है?"},
                1.0, "yesno",
                category="blood: cough",
                symptom="hemoptysis",
                risk_factor=True
            ),
            make_question(
                {"en": "Approximately how much blood comes out each time?",
                "hi": "प्रत्येक बार लगभग कितना खून आता है?"},
                0.8, "text",
                category="hemoptysis_estimate",
                symptom="hemoptysis"
            ),
            
            #({"en": "What gives relief?", "hi": "किस बात से आराम मिलता है?"}, 0.6, "text")
        ])

        cough.add_child(productive_cough)
        cough.add_child(dry_cough)
        cough.add_child(hemoptysis)
        self.symptom_tree_roots["cough"] = cough
        self.symptom_tree_roots["cough < 2 wks"] = cough
        self.symptom_tree_roots["cough > 2 wks"] = cough
        
        # Enhanced Dyspnea Tree
        dyspnea = SymptomNode("dyspnea", [
            
            ({"en": "Did the breathlessness start suddenly or develop gradually?", "hi": "क्या सांस फूलना अचानक शुरू हुआ या धीरे-धीरे बढ़ा?"}, 0.9, "choice"),
            ({"en": "When does shortness of breath usually occur: early morning or at night?", "hi": "सांस लेने में तकलीफ अधिकतर कब होता है: सुबह जल्दी या रात में?"}, 0.7, "choice"),
            
            ({"en": "How much physical effort brings on the breathlessness?", "hi": "कितनी मेहनत करने पर सांस फूलने लगता है?"}, 0.8, "choice"),
            ({"en": "Before the shortness of breath starts, do you have cough with phlegm?", "hi": "सांस लेने में तकलीफ शुरू होने से पहले क्या आपको कफ के साथ खांसी होती है? शुरू होने से पहले क्या आपको कफ के साथ खांसी होती है?"}, 0.7, "yesno"),
            ({"en": "Do you also get chest pain, wheezing, or fever?", "hi": "क्या इसके साथ सीने में दर्द, घरघराहट या बुखार होता है?"}, 0.8, "yesno"),
            
            #({"en": "What gives you relief?", "hi": "किस बात से आपको राहत मिलती है?"}, 0.7, "text")
        ])

        orthopnea = SymptomNode("orthopnea", [
            ({"en": "How many pillows do you need to sleep comfortably?", "hi": "आराम से सोने के लिए आपको कितने तकियों की आवश्यकता होती है?"}, 0.9, "number"),
            ({"en": "For how long have you had this problem?", "hi": "यह समस्या आपको कब से है?"}, 0.7, "number")
        ])
        pnd = SymptomNode("paroxysmal nocturnal dyspnea", [
            ({"en": "Do you wake up at night gasping for air?", "hi": "क्या आप रात में सांस फूलने के कारण हड़बड़ा कर उठते हैं?"}, 0.9, "yesno"),
            #({"en": "How many times per week does this happen?", "hi": "सप्ताह में यह कितनी बार होता है?"}, 0.7, "number"),
            ({"en": "Do you get relief after sitting up?", "hi": "क्या बैठने पर आराम मिलता है?"}, 0.8, "yesno")
        ])

        dyspnea.add_child(orthopnea)
        dyspnea.add_child(pnd)
        self.symptom_tree_roots["Dyspnea"] = dyspnea
        self.symptom_tree_roots["dyspnea"] = dyspnea
        self.symptom_tree_roots["Breathlessness"] = dyspnea
        self.symptom_tree_roots["breathlessness"] = dyspnea

        # Enhanced Chest Pain Tree
        
        chest_pain = SymptomNode("chest pain", [
            
            ({"en": "How often do these chest pain episodes occur?", "hi": "सीने में दर्द की ये घटनाएं कितनी बार होती हैं?"}, 0.7, "number"),
            
            ({"en": "What is the character of the pain: sharp, stabbing, aching, or tight/pressing?", "hi": "दर्द का प्रकार क्या है: तीखा, चुभने जैसा, भारी/दुखता हुआ, या जकड़न/दबाव जैसा?"}, 0.9, "choice"),
            ({"en": "Does the pain travel to the shoulder, arm, jaw, or back?", "hi": "क्या दर्द कंधे, बांह, जबड़े या पीठ तक फैलता है?"}, 0.9, "yesno"),
            
            ({"en": "Does exertion or deep breathing make the pain worse?", "hi": "क्या मेहनत करने या गहरी सांस लेने पर दर्द बढ़ता है?"}, 0.8, "yesno"),
            #({"en": "Do rest or medicines relieve the pain?", "hi": "क्या आराम या दवाइयों से दर्द में राहत मिलती है?"}, 0.8, "yesno"),
            #({"en": "Do you also have night sweats?", "hi": "क्या आपको रात में पसीना भी आता है?"}, 0.6, "yesno")
        ])
        

        palpitations = SymptomNode("palpitation", [
            ({"en": "Do you feel your heartbeat racing, pounding, or skipping?", "hi": "क्या आपको दिल की धड़कन तेज़, ज़ोर-ज़ोर से, या अनियमित महसूस होती है?"}, 0.9, "yesno"),
        ])

        anxiety = SymptomNode("anxiety", [
            ({"en": "Do you experience episodes of anxiety or restlessness?", "hi": "क्या आपको घबराहट या बेचैनी के दौरे आते हैं?"}, 0.9, "yesno"),
        ])

        chest_pain.add_child(orthopnea)
        chest_pain.add_child(dyspnea)
        chest_pain.add_child(palpitations)
        chest_pain.add_child(anxiety)
        self.symptom_tree_roots["chest pain"] = chest_pain
        """

        # Enhanced Abdominal Pain Tree
        """
        abdominal_pain = SymptomNode("abdominal pain", [
            #({"en": "Where is the pain: upper stomach, right upper side, or around the navel?", "hi": "दर्द कहाँ है: ऊपरी पेट, दाहिनी ऊपरी तरफ, या नाभि के आसपास?"}, 0.9, "choice"),
            #({"en": "For how long have you had this abdominal pain?", "hi": "पेट में दर्द आपको कब से है?"}, 0.8, "number"),
            ({"en": "Did the pain start suddenly or develop gradually?", "hi": "क्या दर्द अचानक शुरू हुआ या धीरे-धीरे बढ़ा?"}, 0.9, "choice"),
            ({"en": "Is the pain related to meals (before/after eating)?", "hi": "क्या दर्द खाने से पहले या बाद में बढ़ता या घटता है?"}, 0.7, "choice"),
            ({"en": "What is the nature of the pain: cramping, burning, or colicky?", "hi": "दर्द कैसा है: मरोड़ जैसा, जलन जैसा, या ऐंठन/उठता-बैठता?"}, 0.9, "choice"),
            #({"en": "Is the pain constant or getting worse over time?", "hi": "क्या दर्द लगातार रहता है या समय के साथ बढ़ रहा है?"}, 0.8, "choice"),
            #({"en": "How severe is the pain: mild, moderate, or severe?", "hi": "दर्द कितना तेज़ है: हल्का, मध्यम, या बहुत तेज़?"}, 0.7, "choice"),
            #({"en": "How often does the pain come and go?", "hi": "दर्द कितनी बार आता-जाता है?"}, 0.7, "text"),
            #({"en": "Does the pain shift location or spread to other areas?", "hi": "क्या दर्द अपनी जगह बदलता है या कहीं और फैलता है?"}, 0.8, "yesno"),
            #({"en": "How long does each pain episode last?", "hi": "प्रत्येक एपिसोड कितनी देर तक रहता है?"}, 0.7, "number"),
            ({"en": "Does having food make the stomach pain worse?", "hi": "क्या खाना खाने से पेट दर्द बढ़ जाता है?"}, 0.8, "yesno"),
            #({"en": "Do food, vomiting, or medicines give relief?", "hi": "क्या खाना, उल्टी या दवाइयों से राहत मिलती है?"}, 0.7, "yesno"),
            #({"en": "Do you also have nausea, vomiting, or diarrhea?", "hi": "क्या इसके साथ आपको मतली, उल्टी या दस्त होते हैं?"}, 0.8, "yesno")
        ])
        self.symptom_tree_roots["abdominal pain"] = abdominal_pain
        self.symptom_tree_roots["abdominal discomfort"] = abdominal_pain

        # Enhanced Diarrhea Tree
        diarrhea = SymptomNode("diarrhea", [
            #({"en": "For how long have you had loose stools?", "hi": "आपको दस्त कितने समय से हैं?"}, 0.8, "number"),
            ({"en": "Did the diarrhea start suddenly or gradually?", "hi": "क्या दस्त अचानक शुरू हुए या धीरे-धीरे बढ़े?"}, 0.8, "choice"),
            ({"en": "Is the diarrhea improving or getting worse?", "hi": "क्या दस्त में सुधार हो रहा है या स्थिति बदतर हो रही है?"}, 0.8, "choice"),
            ({"en": "How many times do you pass stools in a day?", "hi": "आप दिन में कितनी बार मल त्याग करते हैं?"}, 0.9, "number"),
            #({"en": "Is it related to meals (before or after eating)?", "hi": "क्या यह खाने से पहले या बाद में बढ़ता या घटता है?"}, 0.6, "choice"),
            #({"en": "Is the quantity small or large each time?", "hi": "हर बार मात्रा कम होती है या ज़्यादा?"}, 0.7, "choice"),
            ({"en": "What is the color of the stools?", "hi": "मल का रंग क्या है?"}, 0.7, "choice"),
            ({"en": "Is there any blood in the stools?", "hi": "क्या मल में खून आता है?"}, 0.9, "yesno"),
            ({"en": "Is there any mucus in the stools?", "hi": "क्या मल में बलगम या चिपचिपा पदार्थ आता है?"}, 0.8, "yesno"),
            ({"en": "What is the consistency: watery or loose?", "hi": "मल की प्रकृति कैसी है: पानी जैसा या ढीला?"}, 0.8, "choice"),
            #({"en": "Do you feel a painful urge to pass stool with little output (tenesmus)?", "hi": "क्या मल त्यागते समय बार-बार दर्दनाक जोर लगाना पड़ता है और कम मात्रा निकलती है?"}, 0.7, "yesno"),
            #({"en": "Do the stools have a foul smell?", "hi": "क्या मल से तेज़ बदबू आती है?"}, 0.6, "yesno"),
            #({"en": "Do the stools float in the toilet?", "hi": "क्या मल पानी में तैरता है?"}, 0.5, "yesno"),
            #({"en": "Do the symptoms worsen with abdominal pain or after eating?", "hi": "क्या पेट दर्द या खाने के बाद लक्षण बढ़ जाते हैं?"}, 0.6, "yesno"),
            #({"en": "Do medicines give relief from the diarrhea?", "hi": "क्या दवाइयों से दस्त में राहत मिलती है?"}, 0.6, "yesno"),
            ({"en": "Do you also have fever or abdominal pain?", "hi": "क्या इसके साथ आपको बुखार या पेट में दर्द है?"}, 0.7, "yesno")
        ])

        vomiting = SymptomNode("vomiting", [
            ({"en": "How many episodes of vomiting did you have?", "hi": "आपको कितनी बार उल्टी हुई?"}, 0.9, "yesno"),
        ])

        diarrhea.add_child(vomiting)
        self.symptom_tree_roots["diarrhea"] = diarrhea
        self.symptom_tree_roots["loose stools"] = diarrhea

        # Enhanced Headache Tree
        headache = SymptomNode("headache", [
            #({"en": "For how long have you had the headache?", "hi": "आपको सिरदर्द कब से है?"}, 0.8, "number"),
            #({"en": "Did the headache start suddenly or develop gradually?", "hi": "क्या सिरदर्द अचानक शुरू हुआ या धीरे-धीरे बढ़ा?"}, 0.9, "choice"),
            ({"en": "Is the headache getting progressively worse?", "hi": "क्या सिरदर्द धीरे-धीरे बढ़ता जा रहा है?"}, 0.8, "yesno"),
            ({"en": "Where is the headache: forehead, temples, or back of the head?", "hi": "सिरदर्द कहाँ होता है: माथे पर, कनपटियों पर, या सिर के पीछे?"}, 0.8, "choice"),
            #({"en": "How severe is it: mild, moderate, or severe?", "hi": "यह कितना तेज़ है: हल्का, मध्यम, या बहुत तेज़?"}, 0.8, "choice"),
            ({"en": "What is the nature of the headache: throbbing, dull, or sharp?", "hi": "सिरदर्द की प्रकृति क्या है: धड़कन, सुस्त या तेज?"}, 0.8, "choice"),
            #({"en": "When is the headache worse: morning, evening, or is it continuous?", "hi": "सिरदर्द कब अधिक होता है: सुबह, शाम या लगातार?"}, 0.7, "choice"),
            #({"en": "Do bright light or loud sounds make the headache worse?", "hi": "क्या तेज़ रोशनी या तेज़ आवाज़ से सिरदर्द बढ़ता है?"}, 0.7, "yesno"),
            #({"en": "Do rest or medicines give relief?", "hi": "क्या आराम या दवाओं से राहत मिलती है?"}, 0.7, "yesno"),
            #({"en": "Do you have migraine-like symptoms (e.g., aura, sensitivity to light)?", "hi": "क्या आपको माइग्रेन जैसे लक्षण होते हैं, जैसे आभा दिखना या रोशनी से तकलीफ़?"}, 0.6, "yesno"),
            #({"en": "Do you also have nausea or vomiting?", "hi": "क्या इसके साथ आपको मतली या उल्टी होती है?"}, 0.7, "yesno"),
            #({"en": "Have you noticed any changes in your vision?", "hi": "क्या आपकी दृष्टि में कोई बदलाव आया है?"}, 0.7, "yesno")
        ])
        self.symptom_tree_roots["headache"] = headache
        self.symptom_tree_roots["severe headache"] = headache

        vomiting = SymptomNode("vomiting", [
            #({"en": "For how long have you been vomiting?", "hi": "आपको उल्टी कब से हो रही है?"}, 0.8, "number"),
            ({"en": "How many times have you vomited?", "hi": "आपको कितनी बार उल्टी हुई है?"}, 0.9, "choice"),
            ({"en": "Is the vomiting getting worse over time?", "hi": "क्या उल्टी समय के साथ बढ़ती जा रही है?"}, 0.8, "yesno"),
            #({"en": "How many episodes of vomiting have you had?", "hi": "उल्टी के कितने एपिसोड हुए हैं?"}, 0.8, "number"),
            #({"en": "Is the vomiting forceful (projectile)?", "hi": "क्या उल्टी तेज़ दबाव से दूर तक निकलती है?"}, 0.6, "choice"),
            #({"en": "At what times does the vomiting usually occur?", "hi": "उल्टी आमतौर पर किस समय होती है?"}, 0.6, "choice")
        ])
        nausea = SymptomNode("nausea", [
            ({"en": "Do you feel nauseated (queasy) even without vomiting?", "hi": "क्या बिना उल्टी के भी आपको मतली या उबकाई महसूस होती है?"}, 0.9, "yesno"),
        ])
        vomiting.add_child(nausea)
        self.symptom_tree_roots["vomiting"] = vomiting

        fatigue = SymptomNode("fatigue", [
            ({"en": "Do you feel tired most of the time?", "hi": "क्या आप अधिकांश समय थकान महसूस करते हैं?"}, 0.8, "choice"),
        ])

        fatigue.add_child(headache)
        """

    
    def build_hybrid_nodes(self):
        """Build hybrid symptom nodes for common combinations with bilingual questions"""
        gi_cluster = HybridSymptomNode(
            self.gi_hybrid_key,
            ["stomach pain", "acidity", "gas", "bloating"],
            [
                (
                    {
                        "en": "Do stomach pain, acidity, gas and bloating usually occur together?",
                        "hi": "क्या पेट दर्द, एसिडिटी, गैस और पेट फूलना आमतौर पर एक साथ होते हैं?",
                        "gu": "શું પેટનું દુખાવું, એસિડિટી, ગેસ અને પેટ ફૂલવું સામાન્ય રીતે એકસાથે થાય છે?",
                        "te": "సాధారణంగా కడుపు నొప్పి, ఆమ్లత్వం, గ్యాస్ మరియు బిగుబాటు అన్నీ ఒకేసారి వస్తాయా?"
                    },
                    0.9,
                    "yesno"
                ),
                (
                    {
                        "en": "Have these stomach issues been continuing for more than a month?",
                        "hi": "क्या ये पेट से जुड़े लक्षण एक महीने से अधिक समय से चल रहे हैं?",
                        "gu": "શું આ પેટની તકલીફો એક મહિનાથી વધુ સમયથી ચાલી રહી છે?",
                        "te": "ఈ కడుపు సమస్యలు ఒక నెలకంటే ఎక్కువ కాలంగా కొనసాగుతున్నాయా?"
                    },
                    0.8,
                    "yesno"
                ),
                (
                    {
                        "en": "Are these symptoms affecting your day-to-day routine?",
                        "hi": "क्या ये लक्षण आपके रोज़मर्रा के कामकाज को प्रभावित कर रहे हैं?",
                        "gu": "શું આ લક્ષણો તમારા રોજિંદા કામકાજને અસર કરી રહ્યા છે?",
                        "te": "ఈ లక్షణాలు మీ రోజువారీ పనులను ప్రభావితం చేస్తున్నాయా?"
                    },
                    0.8,
                    "yesno"
                ),
            ],
            weight=1.3,
        )
        self.hybrid_symptom_nodes[self.gi_hybrid_key] = gi_cluster
        
        # Fever + Cough hybrid
        fever_cough = HybridSymptomNode(
            "Fever_Cough",
            ["fever", "cough"],
            [
                make_question(
                    {
                        "en": "Did the fever and cough begin around the same time?",
                        "hi": "क्या बुखार और खांसी लगभग एक ही समय शुरू हुए थे?",
                        "gu": "શું તાવ અને ઉધરસ લગભગ એક જ સમયે શરૂ થયા હતા?",
                        "te": "జ్వరం మరియు దగ్గు దాదాపు ఒకేసారి ప్రారంభమయ్యాయా?"
                    },
                    0.8, "yesno",
                    category="duration: fever_cough",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "With the fever, do you also feel chest congestion?",
                        "hi": "क्या बुखार के साथ आपके सीने में जमाव महसूस होता है?",
                        "gu": "તાવ સાથે શું તમને છાતીમાં જમાવ અથવા ભારેપણું લાગે છે?",
                        "te": "జ్వరంతో పాటు మీకు ఛాతిలో గట్టిగా ఉన్నట్టు లేదా కఫం గుదికట్టినట్టు అనిపిస్తున్నదా?"
                    },
                    0.9, "yesno",
                    category="chest congestion",
                    symptom="chest congestion"
                ),
                make_question(
                    {
                        "en": "Do you feel chills or shivering with the fever?",
                        "hi": "क्या बुखार के साथ आपको ठंड लगना या कंपकंपी होती है?",
                        "gu": "શું તમને તાવ સાથે થડથડાટી અથવા કંપારી લાગે છે?",
                        "te": "జ్వరంతో పాటు మీకు చలి వణుకు లేదా కంపడం వస్తుందా?"
                    },
                    0.9, "yesno",
                    category="chills",
                    symptom="chills"
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["Fever_Cough"] = fever_cough
        
        # Fever + Headache hybrid
        fever_headache = HybridSymptomNode(
            "Fever_Headache",
            ["fever", "headache"],
            [
                make_question(
                    {
                        "en": "Along with fever and headache, do you also feel nauseated?",
                        "hi": "क्या बुखार और सिरदर्द के साथ आपको मतली भी होती है?",
                        "gu": "તાવ અને માથાના દુખાવા સાથે શું તમને મિતલી પણ લાગે છે?",
                        "te": "జ్వరం మరియు తలనొప్పితో పాటు మీకు వికారం కూడా అనిపిస్తున్నదా?"
                    },
                    0.9, "yesno",
                    category="nausea",
                    symptom="nausea"
                ),
                make_question(
                    {
                        "en": "Did the fever and headache start at about the same time?",
                        "hi": "क्या बुखार और सिरदर्द लगभग एक ही समय शुरू हुए थे?",
                        "gu": "શું તાવ અને માથાનો દુખાવો લગભગ એક જ સમયે શરૂ થયા હતા?",
                        "te": "జ్వరం మరియు తలనొప్పి దాదాపు ఒకేసారి ప్రారంభమయ్యాయా?"
                    },
                    0.8, "yesno",
                    category="duration: fever_headache",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you feel chills or shivering with the fever?",
                        "hi": "क्या बुखार के साथ आपको ठंड लगना या कंपकंपी होती है?",
                        "gu": "શું તમને તાવ સાથે થડથડાટી અથવા કંપારી લાગે છે?",
                        "te": "జ్వరంతో పాటు మీకు చలి వణుకు లేదా కంపడం వస్తుందా?"
                    },
                    0.9, "yesno",
                    category="chills",
                    symptom="chills"
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["fever_headache"] = fever_headache

        # Fever + Vomiting hybrid
        fever_vomiting = HybridSymptomNode(
            "Fever_Vomiting",
            ["fever", "vomiting"],
            [
                make_question(
                    {
                        "en": "When did the fever and vomiting start?",
                        "hi": "बुखार और उल्टी कब शुरू हुई?",
                        "gu": "તાવ અને ઉલ્ટી ક્યારે શરૂ થઈ?",
                        "te": "జ్వరం మరియు వాంతులు ఎప్పుడు ప్రారంభమయ్యాయి?"
                    },
                    0.8, "yesno",
                    category="duration: fever_vomiting",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you feel chills or shivering with the fever?",
                        "hi": "क्या बुखार के साथ आपको ठंड लगना या कंपकंपी होती है?",
                        "gu": "શું તમને તાવ સાથે થડથડાટી અથવા કંપારી લાગે છે?",
                        "te": "జ్వరంతో పాటు మీకు చలి వణుకు లేదా కంపడం వస్తుందా?"
                    },
                    0.9, "yesno",
                    category="chills",
                    symptom="chills"
                ),
                make_question(
                    {
                        "en": "How many times have you vomited?",
                        "hi": "आपको कितनी बार उल्टी हुई है?",
                        "gu": "તમને અત્યાર સુધી કેટલા વખત ઉલ્ટી થઈ છે?",
                        "te": "ఇప్పటివరకు మీరు ఎన్ని సార్లు వాంతి చేశారు?"
                    },
                    0.9, "yesno",
                    category="frequency: vomiting",
                    symptom= None
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["Fever_Vomiting"] = fever_vomiting

        # Headache + Vomiting hybrid
        headache_vomiting = HybridSymptomNode(
            "Headache_Vomiting",
            ["headache", "vomiting"],
            [
                make_question(
                    {
                        "en": "How many times have you vomited?",
                        "hi": "आपको कितनी बार उल्टी हुई है?",
                        "gu": "તમને અત્યાર સુધી કેટલા વખત ઉલ્ટી થઈ છે?",
                        "te": "ఇప్పటివరకు మీరు ఎన్ని సార్లు వాంతి చేశారు?"
                    },
                    0.9, "yesno",
                    category="frequency: vomiting",
                    symptom=None
                ),

                make_question(
                    {
                        "en": "Does vomiting relieve your headache or make it worse?",
                        "hi": "क्या उल्टी करने से सिरदर्द में राहत मिलती है या दर्द बढ़ता है?",
                        "gu": "ઉલ્ટી કરવાથી તમારા માથાના દુખાવામાં રાહત મળે છે કે દુખાવો વધે છે?",
                        "te": "వాంతి చేసినప్పుడు మీ తలనొప్పి తగ్గుతుందా లేదా మరింత పెరుగుతుందా?"
                    },
                    0.9, "yesno",
                    category="post_vomiting_relief",
                    symptom=None
                ),

                make_question(
                    {
                        "en": "Do you experience visual disturbances, dizziness, or sensitivity to light before or during the headache?",
                        "hi": "क्या सिरदर्द से पहले या दौरान आपको चक्कर, रोशनी से परेशानी या धुंधला दिखना महसूस होता है?",
                        "gu": "શું સિરદર્દ પહેલાં અથવા દરમિયાન તમને ચક્કર, પ્રકાશથી તકલીફ અથવા ધુંધળું દેખાવું લાગે છે?",
                        "te": "తలనొప్పి మొదలయ్యే ముందు లేదా సమయంలో మీకు తల తిరగటం, వెలుతురు సహించలేకపోవడం లేదా మసకగా కనిపించడం వంటి సమస్యలు ఉంటాయా?"
                    },
                    0.9, "yesno",
                    category="dizziness",
                    symptom=None
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["Headache_Vomiting"] = headache_vomiting

        # Diarrhea + Vomiting hybrid
        diarrhea_vomiting = HybridSymptomNode(
            "Diarrhea_Vomiting",
            ["diarrhea", "vomiting"],
            [
                make_question(
                    {
                        "en": "When diarrhea and vomiting occur together, do you also feel weak or dehydrated?",
                        "hi": "जब दस्त और उल्टी साथ-साथ हों, तो क्या आप कमजोरी या निर्जलीकरण महसूस करते हैं?",
                        "gu": "જ્યારે દસ્ત અને ઉલ્ટી બંને હોય ત્યારે શું તમે નબળાઈ અથવા ડિહાઇડ્રેશન અનુભવતા હો?",
                        "te": "డయ్యేరియా మరియు వాంతులు రెండూ కలిసి వచ్చినప్పుడు మీకు బలహీనంగా లేదా డీహైడ్రేషన్‌గా అనిపిస్తున్నదా?"
                    },
                    0.9, "yesno",
                    category="dehydration",
                    symptom="dehydration"
                ),
                make_question(
                    {
                        "en": "Did the diarrhea start suddenly or gradually?",
                        "hi": "क्या दस्त अचानक शुरू हुए या धीरे-धीरे बढ़े?",
                        "gu": "દસ્ત અચાનક શરૂ થયા કે ધીમે ધીમે વધ્યા?",
                        "te": "డయ్యేరియా అకస్మాత్తుగా మొదలైందా లేదా క్రమంగా పెరిగిందా?"
                    },
                    0.8, "yesno",
                    category="type: diarrhea",
                    symptom="diarrhea"
                ),
                make_question(
                    {
                        "en": "How many times have you vomited?",
                        "hi": "आपको कितनी बार उल्टी हुई है?",
                        "gu": "તમને અત્યાર સુધી કેટલા વખત ઉલ્ટી થઈ છે?",
                        "te": "ఇప్పటివరకు మీరు ఎన్ని సార్లు వాంతి చేశారు?"
                    },
                    0.9, "yesno",
                    category="frequency: vomiting",
                    symptom=None
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["Diarrhea_Vomiting"] = diarrhea_vomiting

        # Abdominal Pain + Diarrhea hybrid
        abdominal_pain_diarrhea = HybridSymptomNode(
            "Abdominal_Pain_Diarrhea",
            ["stomach pain", "diarrhea"],
            [
                make_question(
                    {
                        "en": "Does the abdominal pain get worse when you have diarrhea, or does it occur at other times?",
                        "hi": "क्या दस्त के समय पेट दर्द बढ़ जाता है, या दर्द अलग समय पर होता है?",
                        "gu": "શું દસ્ત વખતે પેટનો દુખાવો વધી જાય છે કે પછી બીજા સમયે પણ રહે છે?",
                        "te": "డయ్యేరియా ఉన్నప్పుడు కడుపు నొప్పి ఎక్కువ అవుతుందా, లేక ఇతర సమయాల్లో కూడా ఉంటుందా?"
                    },
                    0.9, "yesno",
                    category="relation_pain_to_diarrhea",
                    symptom="stomach pain"
                ),
                make_question(
                    {
                        "en": "Did the diarrhea start suddenly or gradually?",
                        "hi": "क्या दस्त अचानक शुरू हुए या धीरे-धीरे बढ़े?",
                        "gu": "દસ્ત અચાનક શરૂ થયા કે ધીમે ધીમે વધ્યા?",
                        "te": "డయ్యేరియా అకస్మాత్తుగా మొదలైందా లేదా క్రమంగా పెరిగిందా?"
                    },
                    0.8, "yesno",
                    category="type: diarrhea",
                    symptom=None
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["Abdominal_Pain_Diarrhea"] = abdominal_pain_diarrhea


        # Chest Pain + Palpitations hybrid
        chest_pain_palpitations = HybridSymptomNode(
            "Chest_Pain_Palpitations",
            ["chest pain", "palpitations"],
            [
                make_question(
                    {
                        "en": "Do the chest pain and the fast heartbeat happen at the same time?",
                        "hi": "क्या सीने का दर्द और तेज़ धड़कन एक ही समय पर होती हैं?",
                        "gu": "શું છાતીનો દુખાવો અને ધબકારા તેજ થવું એક જ સમયે થાય છે?",
                        "te": "ఛాతీ నొప్పి మరియు గుండె వేగంగా కొట్టుకోవడం ఒకేసారి జరుగుతాయా?"
                    },
                    0.9, "yesno",
                    category="duration: chest_pain_palpitations",
                    symptom=None,
                    risk_factor=True
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["Chest_Pain_Palpitations"] = chest_pain_palpitations

        # Fatigue + Vomiting hybrid
        fatigue_vomiting = HybridSymptomNode(
            "Fatigue_Vomiting",
            ["fatigue", "vomiting"],
            [
                make_question(
                    {
                        "en": "How many times have you vomited?",
                        "hi": "आपको कितनी बार उल्टी हुई है?",
                        "gu": "તમને અત્યાર સુધી કેટલા વખત ઉલ્ટી થઈ છે?",
                        "te": "ఇప్పటివరకు మీరు ఎన్ని సార్లు వాంతి చేశారు?"
                    },
                    0.9, "yesno",
                    category="frequency: vomiting",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you feel tired most of the time?",
                        "hi": "क्या आप अधिकांश समय थकान महसूस करते हैं?",
                        "gu": "શું તમે મોટાભાગના સમયમાં થાક અનુભવતા હો?",
                        "te": "మీకు ఎక్కువ సమయం అలసటగా అనిపిస్తున్నదా?"
                    },
                    0.8, "yesno",
                    category="weakness",
                    symptom="weakness"
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["Fatigue_Vomiting"] = fatigue_vomiting


        '''
        cough_diarrhea = HybridSymptomNode(
            "Cough_Diarrhea",
            ["cough", "diarrhea"],
            [
                make_question(
                    {"en": "Along with your cough illness, do you also have diarrhea?",
                    "hi": "क्या खांसी की बीमारी के साथ आपको दस्त भी हैं?"},
                    0.9, "yesno",
                    category="coexistence: cough_diarrhea",
                    symptom="cough_diarrhea"
                ),
                make_question(
                    {"en": "Do you bring up phlegm (sputum) when you cough?",
                    "hi": "क्या खांसी के साथ बलगम निकलता है?"},
                    0.9, "yesno",
                    category="productive_cough",
                    symptom="sputum"
                ),
                make_question(
                    {"en": "Did the diarrhea start suddenly or gradually?",
                    "hi": "क्या दस्त अचानक शुरू हुए या धीरे-धीरे बढ़े?"},
                    0.8, "yesno",
                    category="onset_pattern: diarrhea",
                    symptom="diarrhea"
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["Cough_Diarrhea"] = cough_diarrhea
        '''
        # Generate additional pairs with bilingual questions
        possible_pairs = {
           
            ("cough", "diarrhea"): {"en": "Does your cough also have phlegm?", "hi": "क्या खांसी के साथ बलगम निकलता है?"},
            
        }

        for (s1, s2), question in possible_pairs.items():
            key = f"{s1}_{s2}".replace(" ", "_")
            hybrid = HybridSymptomNode(
                key,
                [s1, s2],
                [(question, 0.9, "yesno")],
                weight=1.2
            )
            self.hybrid_symptom_nodes[key.lower()] = hybrid
        







        # ──────────────────────────────────────────────────────────────────────────────
        # Triple symptom combinations with bilingual questions
        # Fever + Chills + Cough
        fever_chills_cough = HybridSymptomNode(
            "Fever_Chills_Cough",
            ["fever", "chills", "cough"],
            [
                make_question(
                    {
                        "en": "Did all three symptoms: fever, chills and cough begin around the same time?",
                        "hi": "क्या तीनों लक्षण: बुखार, ठंड लगना और खांसी लगभग एक ही समय पर शुरू हुए?",
                        "gu": "શું ત્રણે લક્ષણો: તાવ, ઠંડ લાગવી અને ઊધરસ લગભગ એક જ સમયે શરૂ થયા હતા?",
                        "te": "జ్వరం, వణుకు (చలి) మరియు దగ్గు మూడు లక్షణాలు దాదాపు ఒకేసారి మొదలయ్యాయా?"
                    },
                    0.9, "yesno",
                    category="onset_sync: fever_chills_cough",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you feel chills or shivering with the fever?",
                        "hi": "क्या बुखार के साथ आपको ठंड लगना या कंपकंपी होती है?",
                        "gu": "શું તમને તાવ સાથે ઠંડ લાગવી કે થડથડાટી થાય છે?",
                        "te": "జ్వరంతో పాటు మీకు చలి వణుకు లేదా కంపడం వస్తుందా?"
                    },
                    0.9, "yesno",
                    category="chills",
                    symptom="chills"
                ),
                make_question(
                    {
                        "en": "Do you bring up phlegm (sputum) when you cough?",
                        "hi": "क्या खांसी के साथ बलगम निकलता है?",
                        "gu": "શું ઉધરસ આવે ત્યારે કફ (કફ/થૂંક) બહાર આવે છે?",
                        "te": "మీరు దగ్గుతున్నప్పుడు కఫం (చీము) వస్తుందా?"
                    },
                    0.9, "yesno",
                    category="phlegm",
                    symptom="phlegm"
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["Fever_Chills_Cough"] = fever_chills_cough

        # ──────────────────────────────────────────────────────────────────────────────
        # Fever + Headache + Vomiting
        fever_headache_vomiting = HybridSymptomNode(
            "Fever_Headache_Vomiting",
            ["fever", "headache", "vomiting"],
            [
                make_question(
                    {
                        "en": "Are fever, headache, and vomiting occurring together—possibly with neck stiffness?",
                        "hi": "क्या बुखार, सिरदर्द और उल्टी एक साथ हो रहे हैं—क्या गर्दन में अकड़न भी है?",
                        "gu": "શું તાવ, માથાનો દુખાવો અને ઉલ્ટી એકસાથે થઈ રહ્યા છે—અને ગળામાં કસરિયાપણું (જમાવ) પણ છે?",
                        "te": "జ్వరం, తలనొప్పి, వాంతులు కలిసి వస్తున్నాయా—తోడుగా మెడ గట్టిపడటం కూడా ఉందా?"
                    },
                    0.95, "yesno",
                    category="neck_stiffness",
                    symptom=None,
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "Did the headache and vomiting start after the fever began?",
                        "hi": "क्या बुखार शुरू होने के बाद सिरदर्द और उल्टी शुरू हुए?",
                        "gu": "શું તાવ શરૂ થયા પછી માથાનો દુખાવો અને ઉલ્ટી શરૂ થઈ?",
                        "te": "జ్వరం మొదలైన తర్వాతే తలనొప్పి మరియు వాంతులు ప్రారంభమయ్యాయా?"
                    },
                    0.85, "yesno",
                    category="duration: fever_headache_vomiting",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you feel chills or shivering with the fever?",
                        "hi": "क्या बुखार के साथ आपको ठंड लगना या कंपकंपी होती है?",
                        "gu": "શું તમને તાવ સાથે ઠંડ લાગવી કે થડથડાટી થાય છે?",
                        "te": "జ్వరంతో పాటు మీకు చలి వణుకు లేదా కంపడం వస్తుందా?"
                    },
                    0.9, "yesno",
                    category="chills",
                    symptom="chills"
                ),
            ],
            weight=1.25
        )
        self.hybrid_symptom_nodes["Fever_Headache_Vomiting"] = fever_headache_vomiting

        # ──────────────────────────────────────────────────────────────────────────────
        # Cough + Dyspnea + Chest Pain
        cough_dyspnea_chestpain = HybridSymptomNode(
            "Cough_Dyspnea_ChestPain",
            ["cough", "dyspnea", "chest pain"],
            [
                make_question(
                    {
                        "en": "Do cough, breathlessness, and chest pain occur together during the same episodes?",
                        "hi": "क्या खांसी, सांस फूलना और सीने का दर्द एक ही एपिसोड में साथ-साथ होते हैं?",
                        "gu": "શું ઊધરસ, શ્વાસ ફુલાવું અને છાતીનો દુખાવો એક જ એપિસોડ દરમિયાન સાથે થાય છે?",
                        "te": "దగ్గు, ఊపిరి తిత్తికబెట్టడం (శ్వాస తీసుకోవడంలో ఇబ్బంది) మరియు ఛాతీ నొప్పి ఒకే సమయంలో ఒకే ఎపిసోడ్‌లో వస్తున్నాయా?"
                    },
                    0.95, "yesno",
                    category="episode_concurrency: resp_triads",
                    symptom=None,
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "When you cough and feel breathless, does the chest pain get worse?",
                        "hi": "जब खांसी आती है और सांस फूलती है, तो क्या सीने का दर्द बढ़ जाता है?",
                        "gu": "જ્યારે તમને ઊધરસ આવે અને શ્વાસ ફૂલતો હોય ત્યારે શું છાતીનો દુખાવો વધી જાય છે?",
                        "te": "మీకు దగ్గు వచ్చినప్పుడు, ఊపిరి బిగుసుకున్నప్పుడు ఛాతి నొప్పి ఎక్కువ అవుతుందా?"
                    },
                    0.9, "yesno",
                    category="pain_worse_with_cough_dyspnea",
                    symptom=None,
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "Do you bring up phlegm (sputum) when you cough?",
                        "hi": "क्या खांसी के साथ बलगम निकलता है?",
                        "gu": "શું ઉધરસ આવે ત્યારે કફ (થૂંક) બહાર આવે છે?",
                        "te": "మీరు దగ్గుతున్నప్పుడు కఫం (చీము) వస్తుందా?"
                    },
                    0.85, "yesno",
                    category="phlegm",
                    symptom="phlegm"
                ),
            ],
            weight=1.3
        )
        self.hybrid_symptom_nodes["Cough_Dyspnea_ChestPain"] = cough_dyspnea_chestpain

        # ──────────────────────────────────────────────────────────────────────────────
        # Abdominal Pain + Diarrhea + Vomiting
        abdpain_diarrhea_vomiting = HybridSymptomNode(
            "AbdominalPain_Diarrhea_Vomiting",
            ["stomach pain", "diarrhea", "vomiting"],
            [
                make_question(
                    {
                        "en": "Do abdominal pain, diarrhea, and vomiting occur together in the same episode?",
                        "hi": "क्या पेट दर्द, दस्त और उल्टी एक ही एपिसोड में साथ-साथ होते हैं?",
                        "gu": "શું પેટનો દુખાવો, દસ્ત અને ઉલ્ટી એક જ એપિસોડમાં સાથે થાય છે?",
                        "te": "కడుపు నొప్పి, డయ్యేరియా, వాంతులు ఒకే ఎపిసోడ్‌లో కలసి వస్తున్నాయా?"
                    },
                    0.95, "yesno",
                    category="episode_concurrency: gi_triads",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do these symptoms appear after eating food that may have been unsafe?",
                        "hi": "क्या ये लक्षण संदिग्ध या असुरक्षित भोजन खाने के बाद दिखाई देते हैं?",
                        "gu": "શું આ લક્ષણો શંકાસ્પદ કે અસ્વચ્છ ખોરાક ખાધા પછી દેખાય છે?",
                        "te": "సరిగా కాకపోయిన / పాడై ఉండొచ్చని అనిపించే ఆహారం తిన్న తర్వాత ఈ లక్షణాలు వస్తున్నాయా?"
                    },
                    0.9, "yesno",
                    category="diet: eat",
                    symptom=None
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["AbdominalPain_Diarrhea_Vomiting"] = abdpain_diarrhea_vomiting

        # ──────────────────────────────────────────────────────────────────────────────
        # Fever + Fatigue + Cough
        fever_fatigue_cough = HybridSymptomNode(
            "Fever_Fatigue_Cough",
            ["fever", "fatigue", "cough"],
            [
                make_question(
                    {
                        "en": "Do you have fever, fatigue, and cough together—like a flu-type illness?",
                        "hi": "क्या आपको बुखार, थकान और खांसी एक साथ हैं—जैसे फ्लू में होता है?",
                        "gu": "શું તમને તાવ, થાક અને ઊધરસ એકસાથે છે—જેમ ફ્લૂમાં થાય છે?",
                        "te": "జ్వరం, అలసట, దగ్గు అన్నీ కలిసి ఫ్లూ వచ్చినట్టు ఉన్నాయా?"
                    },
                    0.9, "yesno",
                    category="flu",
                    symptom="flu"
                ),
                make_question(
                    {
                        "en": "Did the fatigue get worse after the fever and cough started?",
                        "hi": "क्या बुखार और खांसी शुरू होने के बाद थकान बढ़ गई?",
                        "gu": "શું તાવ અને ઊધરસ શરૂ થયા પછી થાક વધુ વધી ગયો છે?",
                        "te": "జ్వరం, దగ్గు మొదలైన తర్వాత మీ అలసట పెరిగిందా?"
                    },
                    0.85, "yesno",
                    category="type: weakness",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you feel chills or shivering with the fever?",
                        "hi": "क्या बुखार के साथ आपको ठंड लगना या कंपकंपी होती है?",
                        "gu": "શું તમને તાવ સાથે ઠંડ લાગવી કે થડથડાટી થાય છે?",
                        "te": "జ్వరంతో పాటు మీకు చలి వణుకు లేదా కంపడం వస్తుందా?"
                    },
                    0.85, "yesno",
                    category="chills",
                    symptom="chills"
                ),
                make_question(
                    {
                        "en": "Do you bring up phlegm (sputum) when you cough?",
                        "hi": "क्या खांसी के साथ बलगम निकलता है?",
                        "gu": "શું ઉધરસ આવે ત્યારે કફ (થૂંક) બહાર આવે છે?",
                        "te": "మీరు దగ్గుతున్నప్పుడు కఫం (చీము) వస్తుందా?"
                    },
                    0.85, "yesno",
                    category="phlegm",
                    symptom="phlegm"
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["Fever_Fatigue_Cough"] = fever_fatigue_cough

        # ──────────────────────────────────────────────────────────────────────────────
        # Headache + Nausea + Vomiting
        headache_nausea_vomiting = HybridSymptomNode(
            "Headache_Nausea_Vomiting",
            ["headache", "nausea", "vomiting"],
            [
                make_question(
                    {
                        "en": "During an episode, do headache, nausea, and vomiting occur together?",
                        "hi": "क्या एक एपिसोड के दौरान सिरदर्द, मतली और उल्टी साथ-साथ होती हैं?",
                        "gu": "શું એક એપિસોડ દરમિયાન માથાનો દુખાવો, મિતલી અને ઉલ્ટી એકસાથે થાય છે?",
                        "te": "ఒక ఎపిసోడ్ సమయంలో తలనొప్పి, వికారం, వాంతులు ఒకేసారి వస్తున్నాయా?"
                    },
                    0.95, "yesno",
                    category="frequency: neuro_gi",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "When you have a headache, does nausea usually come before vomiting?",
                        "hi": "जब सिरदर्द होता है, तो क्या उल्टी से पहले आमतौर पर मतली होती है?",
                        "gu": "જ્યારે તમને માથાનો દુખાવો હોય ત્યારે શું સામાન્ય રીતે ઉલ્ટી પહેલાં મિતલી થાય છે?",
                        "te": "మీకు తలనొప్పి ఉన్నప్పుడు సాధారణంగా వాంతి చేసుకునే ముందు వికారం వస్తుందా?"
                    },
                    0.9, "yesno",
                    category="sequence_nausea_before_vomit",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "How many times have you vomited?",
                        "hi": "आपको कितनी बार उल्टी हुई है?",
                        "gu": "તમને અત્યાર સુધી કેટકેટલી વાર ઉલ્ટી થઈ છે?",
                        "te": "ఇప్పటివరకు మీరు ఎన్ని సార్లు వాంతి చేశారు?"
                    },
                    0.9, "yesno",
                    category="frequency: vomiting",
                    symptom=None
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["Headache_Nausea_Vomiting"] = headache_nausea_vomiting

        # 1) Acidity_Backpain_gas
        acidity_backpain_gas = HybridSymptomNode(
            "Acidity_Backpain_gas",
            ["acidity", "back pain", "gas"],
            [
                make_question(
                    {
                        "en": "Can you describe the nature of the pain or discomfort— is it burning, dull, sharp, or pressure-like?",
                        "hi": "दर्द या असहजता का स्वरूप बताएं— क्या यह जलन जैसा, हल्का, तेज़ चुभन जैसा या दबाव जैसा है?",
                        "gu": "દર્દ અથવા અસ્વસ્થતા કેવી છે તે જણાવો— શું તે સળવળાટ જેવી, થોડી ધીમી, તિક્ષ્ણ ચભચભતી કે દબાણ જેવી છે?",
                        "te": "నొప్పి లేదా అసౌకర్యం ఎలా ఉంటుంది— అది మంటలా, మామూలు మంటగా, గుచ్చినట్టు లేదా ఒత్తిడి ఉన్నట్టు అనిపిస్తుందా?"
                    },
                    0.9, "yesno",
                    category="pain: burning",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "When do these symptoms usually occur: after meals, on an empty stomach, or at night? Do certain foods or positions make it worse or better?",
                        "hi": "ये लक्षण आम तौर पर कब होते हैं: खाने के बाद, खाली पेट, या रात में? क्या कुछ भोजन या शरीर की स्थिति से यह बढ़ते या घटते हैं?",
                        "gu": "આ લક્ષણો સામાન્ય રીતે ક્યારે થાય છે: ભોજન પછી, ખાલી પેટે કે રાત્રે? શું કંઈ ખાસ ખોરાક અથવા શરીરની પોઝિશનથી આ વધે કે ઘટે છે?",
                        "te": "ఈ లక్షణాలు సాధారణంగా ఎప్పుడు వస్తాయి: భోజనం తర్వాత, ఖాళీ కడుపుతో, లేదా రాత్రిళ్ళు? కొన్ని ఆహారాలు లేదా శరీర స్థితులు వీటిని ఎక్కువగా చేస్తాయా లేదా తగ్గిస్తాయా?"
                    },
                    0.9, "yesno",
                    category="diet: at",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Does the pain move anywhere: for example, toward the chest, shoulder, or down the back?",
                        "hi": "क्या दर्द कहीं फैलता/सरकता है— जैसे छाती, कंधे या पीठ के निचले हिस्से की ओर?",
                        "gu": "શું દુખાવો ક્યાંક ખસે છે— જેમ કે છાતી, ખભા અથવા નીચેની પીઠ તરફ?",
                        "te": "నొప్పి ఎక్కడికైనా విస్తరిస్తుందా— ఉదాహరణకు ఛాతీకి, భుజానికి లేదా వెన్ను కింద భాగానికి?"
                    },
                    0.85, "yesno",
                    category="location: where",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you have nausea, vomiting, bloating, blood in vomit/stool, weight loss, or difficulty swallowing?",
                        "hi": "क्या आपको मितली, उल्टी, पेट फूला हुआ महसूस होना, उल्टी/मल में खून, वजन घटना, या निगलने में कठिनाई है?",
                        "gu": "શું તમને મિતલી, ઉલ્ટી, પેટ ફૂલવું, ઉલ્ટી/મલમાં લોહી આવવું, વજન ઘટવું અથવા ગળવાથી ગળે નહિં તેવું લાગે છે?",
                        "te": "మీకు వికారం, వాంతులు, పొత్తికడుపు బిగుసుకోవడం, వాంతి/మలంలో రక్తం, బరువు తగ్గడం లేదా మింగటం కష్టంగా ఉందా?"
                    },
                    0.95, "yesno",
                    category="nausea",
                    symptom="nausea",
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "Do you consume alcohol, smoke, or take painkillers regularly?",
                        "hi": "क्या आप शराब पीते हैं, धूम्रपान करते हैं, या नियमित रूप से दर्द निवारक दवाएं लेते हैं?",
                        "gu": "શું તમે દારૂ પીવો છો, ધુમ્રપાન કરો છો અથવા નિયમિત રીતે પેનકિલર દવાઓ લો છો?",
                        "te": "మీరు ఆల్కహాల్ తాగుతారా, పొగ త్రాగుతారా లేదా నిత్యం నొప్పి నివారక మందులు తీసుకుంటున్నారా?"
                    },
                    0.85, "yesno",
                    category="alcohol",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Any history of hypertension, diabetes, or heart disease?",
                        "hi": "क्या उच्च रक्तचाप, मधुमेह या हृदय रोग का कोई इतिहास है?",
                        "gu": "શું તમને હાઈ બ્લડ પ્રેશર, ડાયાબિટીસ અથવા હૃદય રોગનો ઇતિહાસ છે?",
                        "te": "మీకు హై బ్లడ్ ప్రెజర్, మధుమేహం లేదా గుండె సంబంధిత వ్యాధుల చరిత్ర ఉందా?"
                    },
                    0.85, "yesno",
                    category="comorbids_cardio_metabolic",
                    symptom=None
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["acidity_backpain_gas"] = acidity_backpain_gas

        # 2) acidity_balance problem_dizziness
        acidity_balance_problem_dizziness = HybridSymptomNode(
            "acidity_balance problem_dizziness",
            ["acidity", "balance problem", "dizziness"],
            [
                make_question(
                    {
                        "en": "Can you describe the dizziness: is it spinning, lightheadedness, or feeling of imbalance?",
                        "hi": "चक्कर का प्रकार बताएं— क्या यह घूमने जैसा, हल्का-सा चकराना, या असंतुलन जैसा महसूस होता है?",
                        "gu": "ચક્કર કેવાં લાગે છે જણાવો— શું તે ચક્કર આવવું (સબ ગૂમે તેવું), હલકું ચકરાવું કે અસ્થિરતા/સંતુલન જતું હોય તેવું લાગે છે?",
                        "te": "మీ తల తిరుగుడు ఎలా ఉంటుంది— అన్నీ తిరిగినట్టు (స్పినింగ్), తేలికగా తల తిరిగినట్టు, లేక స్థిరంగా నిలబడలేకపోయినట్టు అనిపిస్తుందా?"
                    },
                    0.9, "yesno",
                    category="type: dizziness",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "When do these symptoms usually occur: after meals, on an empty stomach, on standing up, or at rest?",
                        "hi": "ये लक्षण कब होते हैं: खाने के बाद, खाली पेट, खड़े होने पर, या आराम की स्थिति में?",
                        "gu": "આ લક્ષણો ક્યારે થાય છે: ભોજન પછી, ખાલી પેટે, ઊભા થતી વખતે કે બિલકુલ આરામમાં?",
                        "te": "ఈ లక్షణాలు ఎప్పుడు ఎక్కువగా వస్తాయి: భోజనం తర్వాతా, ఖాళీ కడుపుతోనా, లేచినప్పుడు (నిలబడినప్పుడు)నా, లేక విశ్రాంతిలో ఉన్నప్పుడు?"
                    },
                    0.9, "yesno",
                    category="onset: dizziness",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you notice nausea, vomiting, ringing in the ears, hearing loss, or visual changes along with dizziness?",
                        "hi": "क्या चक्कर के साथ मितली, उल्टी, कान में घंटी बजना, सुनने में कमी, या दृष्टि में बदलाव होता है?",
                        "gu": "શું ચક્કર સાથે તમને મિતલી, ઉલ્ટી, કાનમાં અવાજ (ઘંટી) વાગવો, સાંભળવામાં ઘટાડો કે દેખવામાં ફેરફાર થાય છે?",
                        "te": "తల తిరుగుడుతో పాటు మీకు వికారం, వాంతులు, చెవిలో మోగుడు, వినికిలో తగ్గుదల లేదా చూపులో మార్పులు ఉన్నాయా?"
                    },
                    0.9, "yesno",
                    category="nausea",
                    symptom="nausea"
                ),
                make_question(
                    {
                        "en": "Have you fainted, felt palpitations, chest pain, or sweating during these episodes?",
                        "hi": "क्या ऐसे एपिसोड में बेहोशी, धड़कन तेज़ होना, सीने में दर्द या पसीना आना होता है?",
                        "gu": "શું આવા એપિસોડ દરમિયાન તમને બેભાન થવું, ધબકારા તેજ થવા, છાતીમાં દુખાવો કે વધુ પસીનો આવવો અનુભવાય છે?",
                        "te": "ఈ ఎపిసోడ్‌ల సమయంలో మీకు జ్ఞానం కోల్పోవటం (సొమ్మసిల్లిపోవటం), గుండె వేగంగా కొట్టుకోవటం, ఛాతి నొప్పి లేదా చెమటలు పట్టడం జరుగుతున్నాయా?"
                    },
                    0.95, "yesno",
                    category="fainting",
                    symptom=None,
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "Do you take any regular medicines or consume alcohol?",
                        "hi": "क्या आप कोई नियमित दवाएं लेते हैं या शराब का सेवन करते हैं?",
                        "gu": "શું તમે કોઈ નિયમિત દવાઓ લો છો અથવા દારૂ પીવો છો?",
                        "te": "మీరు ఏవైనా మందులు తరచుగా తీసుకుంటున్నారా లేదా ఆల్కహాల్ సేవిస్తున్నారా?"
                    },
                    0.85, "yesno",
                    category="alcohol",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Have you noticed weight loss, black stools, weakness, or pallor recently?",
                        "hi": "क्या हाल में वजन घटा है, काला मल आया है, कमजोरी या पीलापन महसूस हुआ है?",
                        "gu": "શું હમણાં વજન ઘટ્યું છે, કાળો મલ આવ્યો છે, કમજોરી કે પીળાશ અનુભવાઈ છે?",
                        "te": "ఇటీవల మీ బరువు తగ్గిందా, నల్లటి మలము, బలహీనత లేదా రంగు పచ్చబడినట్టు (పచ్చని వర్ణం/పేలర్) కనిపిస్తున్నదా?"
                    },
                    0.95, "yesno",
                    category="weakness",
                    symptom=None,
                    risk_factor=True
                ),
            ],
            weight=1.25
        )
        self.hybrid_symptom_nodes["acidity_balance problem_dizziness"] = acidity_balance_problem_dizziness

        # 3) chest pain_irregular heartbeat_rapid breathing
        chest_pain_irregular_heartbeat_rapid_breathing = HybridSymptomNode(
            "chest pain_irregular heartbeat_rapid breathing",
            ["chest pain", "irregular heartbeat", "rapid breathing"],
            [
                make_question(
                    {
                        "en": "Can you describe the chest pain? Is it pressure-like, sharp, or burning? Where exactly do you feel it?",
                        "hi": "सीने के दर्द का वर्णन करें— क्या यह दबाव जैसा, तेज़ चुभन जैसा या जलन जैसा है? दर्द ठीक कहाँ होता है?",
                        "gu": "છાતીનો દુખાવો કેવો લાગે છે— શું તે દબાણ જેવો, તીખો ચોખ્ખો કે સળવળાટ જેવો છે? દુખાવો ચોક્કસ ક્યાં લાગે છે?",
                        "te": "ఛాతీ నొప్పి ఎలా ఉంటుంది— ఒత్తిడి ఉన్నట్టా, గుచ్చినట్టా, మంటలాగా ఉండటామా? ఆ నొప్పి కచ్చితంగా ఎక్కడ ఉంది?"
                    },
                    0.95, "yesno",
                    category="type: chest pain",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Does the chest pain radiate anywhere— to the arm, jaw, neck, shoulder, or back?",
                        "hi": "क्या सीने का दर्द कहीं और जाता है— जैसे बांह, जबड़े, गर्दन, कंधे या पीठ की ओर?",
                        "gu": "શું છાતીનો દુખાવો હાથ, જડબા, ગળા, ખભા કે પીઠ તરફ ફેલાય છે?",
                        "te": "ఛాతీ నొప్పి చేతి, దవడ, మెడ, భుజం లేదా వెన్ను వైపు విస్తరిస్తుందా?"
                    },
                    0.95, "yesno",
                    category="location: chest pain",
                    symptom=None,
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "When did the pain start, how long does it last, and what brings it on— exertion, rest, emotion, or posture?",
                        "hi": "दर्द कब शुरू हुआ, कितनी देर रहता है, और इसे क्या ट्रिगर करता है— मेहनत, आराम, भावनात्मक तनाव या शरीर की स्थिति?",
                        "gu": "દુખાવો ક્યારે શરૂ થયો, કેટલો સમય રહે છે, અને એને શું ટ્રિગર કરે છે— મહેનત, આરામ, લાગણીજન્ય તણાવ કે શરીરની સ્થિતિ?",
                        "te": "నొప్పి ఎప్పుడు ప్రారంభమైంది, ఎంతసేపు ఉంటుంది, దాన్ని ఏం ట్రిగర్ చేస్తుంది— శ్రమ, విశ్రాంతి, భావోద్వేగాలు, లేక శరీర స్థితి?"
                    },
                    0.95, "yesno",
                    category="duration: chest pain",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you feel your heartbeat racing or skipping? Have you experienced dizziness, fainting, or blackouts?",
                        "hi": "क्या धड़कन तेज़/अनियमित लगती है? क्या चक्कर, बेहोशी या ब्लैकआउट हुआ है?",
                        "gu": "શું તમને ધબકારો ઘણો તેજ અથવા ઉછળતો-છૂટતો લાગે છે? શું તમને ચક્કર, બેભાન થવું અથવા બ્લેકઆઉટ આવ્યો છે?",
                        "te": "మీ గుండె బాగా వేగంగా కొట్టుకోవడం లేదా దెబ్బలు మిస్ అయినట్టు అనిపిస్తుందా? మీకు తల తిరగటం, మూర్చపోవటం, బ్లాక్ అవుట్ అనుభవించారా?"
                    },
                    0.95, "yesno",
                    category="dizziness",
                    symptom="dizziness",
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "Are you breathless all the time or suddenly? Any cough, blood with cough, leg swelling, or recent long travel/immobilization?",
                        "hi": "क्या सांस लगातार फूलती है या अचानक? खांसी, खांसी में खून, पैरों में सूजन, या हाल में लंबी यात्रा/अचल रहना हुआ है?",
                        "gu": "શું તમને સતત અથવા અચાનક શ્વાસ ફૂલવાનો અનુભવ થાય છે? ઊધરસ, ઉધરસમાં લોહી, પગોમાં સૂજન અથવા તાજેતરમાં લાંબી મુસાફરી/ઘણો સમય સુતા/બેસતા રહ્યા છો?",
                        "te": "మీకు ఎప్పుడూ లేదా అకస్మాత్తుగా ఊపిరి బిగుసుకుపోవడం ఉందా? దగ్గు, దగ్గుతో రక్తం, కాళ్ల వాపు, లేదా ఇటీవలి కాలంలో ఎక్కువ సేపు ప్రయాణం/నిశ్చలంగా ఉండటం జరిగిందా?"
                    },
                    0.95, "yesno",
                    category="shortness of breath",
                    symptom="shortness of breath",
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "Do you have risk factors such as hypertension, diabetes, smoking, thyroid disease, stimulant drug use, or recent emotional stress?",
                        "hi": "क्या उच्च रक्तचाप, मधुमेह, धूम्रपान, थायरॉइड रोग, उत्तेजक दवाओं का उपयोग, या हाल का भावनात्मक तनाव जैसे जोखिम कारक हैं?",
                        "gu": "શું તમને હાઈ બ્લડ પ્રેશર, ડાયાબિટીસ, ધુમ્રપાન, થાયરોઇડ રોગ, સ્ટિમ્યુલન્ટ દવાઓનો ઉપયોગ અથવા તાજેતરમાં ભાવનાત્મક તણાવ જેવા જોખમકારક તત્વો છે?",
                        "te": "మీకు హై బిపి, షుగర్, పొగ త్రాగడం, థైరాయిడ్ వ్యాధి, స్టిమ్యులెంట్ మందుల వాడకం లేదా ఇటీవలి భావోద్వేగ ఒత్తిడి వంటి ప్రమాదకారకాలు ఉన్నాయా?"
                    },
                    0.9, "yesno",
                    category="diabetes",
                    symptom="diabetes"
                ),
            ],
            weight=1.35
        )
        self.hybrid_symptom_nodes["chest pain_irregular heartbeat_rapid breathing"] = chest_pain_irregular_heartbeat_rapid_breathing

        # 4) Urine issue_Kidney issue
        urine_issue_kidney_issue = HybridSymptomNode(
            "Urine issue_Kidney issue",
            ["urine issue", "kidney issue"],
            [
                make_question(
                    {
                        "en": "Can you describe the urinary problem: is it burning, painful, frequent, or difficult to pass urine?",
                        "hi": "मूत्र की समस्या बताएं— क्या जलन, दर्द, बार-बार पेशाब आना, या पेशाब करने में कठिनाई है?",
                        "gu": "મૂત્ર સંબંધિત તકલીફ કેવી છે જણાવો— શું મૂત્ર કરતી વખતે સળવળાટ, દુખાવો, વારંવાર મૂત્ર થવું કે મૂત્ર કરવા મુશ્કેલી થાય છે?",
                        "te": "మీ మూత్ర సమస్యను వివరించండి— మూత్రం పోయేటప్పుడు మంటా, నొప్పి, తరచుగా మూత్రం పోవాలనిపించడమా, లేక మూత్రం చేసేందుకు కష్టంగా ఉందా?"
                    },
                    0.9, "yesno",
                    category="type: urine issue",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Have you noticed any change in the color, smell, or amount of urine—for example, blood, frothiness, or dark urine?",
                        "hi": "क्या मूत्र के रंग, गंध या मात्रा में कोई बदलाव देखा है— जैसे खून, झागदार या गहरे रंग का मूत्र?",
                        "gu": "શું તમે મૂત્રના રંગ, વાસ અથવા માત્રામાં ફેરફાર જોયો છે— જેમ કે લોહીવાળું, વધુ ઝગમગતું/ફીણવાળું અથવા ગાઢ રંગનું મૂત્ર?",
                        "te": "మీ మూత్రం రంగు, వాసన, పరిమాణంలో ఏమైనా మార్పు గమనించారా— ఉదాహరణకు రక్తం కలిసినట్టు, నురుగు రావడం లేదా చాలా ముదురు రంగు?"
                    },
                    0.9, "yesno",
                    category="urine color",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you have swelling of the feet, puffiness around the eyes, or decrease in urine output?",
                        "hi": "क्या पैरों में सूजन, आंखों के आसपास फूलापन, या मूत्र की मात्रा में कमी है?",
                        "gu": "શું તમારા પગમાં સૂજન, આંખોના આસપાસ ફૂલેલુંપણું અથવા મૂત્રની માત્રામાં ઘટાડો છે?",
                        "te": "మీ కాళ్లలో వాపు, కళ్ల చుట్టూ ఉబ్బర, లేదా మూత్రం పరిమాణం తగ్గిపోవడం ఉన్నాయా?"
                    },
                    0.95, "yesno",
                    category="foot swelling",
                    symptom="foot swelling",
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "Have you experienced fever, chills, or flank pain radiating to the groin?",
                        "hi": "क्या बुखार, ठंड लगना, या बगल/पीठ के पास दर्द जो जांघ के जोड़ तक जाता है, हुआ है?",
                        "gu": "શું તમને તાવ, ઠંડ લાગવી અથવા બાજુ/પીઠ તરફનો દુખાવો જે છેડી (ગ્રોઇન) સુધી જાય છે, થયો છે?",
                        "te": "మీకు జ్వరం, చలి వణుకు, లేదా నడుము/పక్క భాగంలో నొప్పి తొడల మధ్య భాగం (గ్రోయిన్) వరకు వెళ్తుందా?"
                    },
                    0.95, "yesno",
                    category="fever",
                    symptom="fever",
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "Do you have a history of diabetes, hypertension, kidney stones, or recent use of painkillers or herbal medicines?",
                        "hi": "क्या मधुमेह, उच्च रक्तचाप, किडनी स्टोन का इतिहास है, या हाल में दर्द निवारक/हर्बल दवाओं का उपयोग किया है?",
                        "gu": "શું તમને ડાયાબિટીસ, હાઈ બ્લડ પ્રેશર, કિડની સ્ટોનનો ઇતિહાસ છે, અથવા તાજેતરમાં પેનકિલર/હર્બલ દવાઓ લીધી છે?",
                        "te": "మీకు షుగర్, హై బిపి, మూత్రపిండ రాళ్లు ఉన్న చరిత్ర ఉందా, లేదా ఇటీవలి కాలంలో నొప్పి మందులు/హెర్బల్ మందులు తరచుగా వాడుతున్నారా?"
                    },
                    0.9, "yesno",
                    category="diabetes",
                    symptom="diabetes"
                ),
                make_question(
                    {
                        "en": "Have you noticed nausea, vomiting, loss of appetite, or fatigue recently?",
                        "hi": "क्या हाल में मितली, उल्टी, भूख कम लगना, या थकान महसूस हुई है?",
                        "gu": "શું હમણાં મિતલી, ઉલ્ટી, ભૂખમાં ઘટાડો અથવા થાક લાગવો નોંધ્યો છે?",
                        "te": "ఇటీవలి కాలంలో మీకు వికారం, వాంతులు, ఆకలి తగ్గడం లేదా అలసటగా ఉండటం గమనించారా?"
                    },
                    0.9, "yesno",
                    category="nausea",
                    symptom="nausea"
                ),
            ],
            weight=1.25
        )
        self.hybrid_symptom_nodes["urine issue_kidney issue"] = urine_issue_kidney_issue

        # 5) Loss of appetite_Less hungry
        loss_of_appetite_less_hungry = HybridSymptomNode(
            "Loss of appetite_Less hungry",
            ["loss of appetite", "less hungry"],
            [
                make_question(
                    {
                        "en": "Since when have you noticed a decrease in appetite: was it sudden or gradual?",
                        "hi": "भूख कम कब से लग रही है— क्या यह अचानक शुरू हुई या धीरे-धीरे?",
                        "gu": "ભૂખમાં ઘટાડો તમને ક્યારેથી લાગ્યો— શું તે અચાનક શરૂ થયો કે ધીમે ધીમે?",
                        "te": "మీ ఆకలి ఎప్పటి నుంచి తగ్గింది— అది ఒక్కసారిగా తగ్గిందా లేదా నెమ్మదిగా క్రమంగా తగ్గిందా?"
                    },
                    0.9, "yesno",
                    category="onset_trend_appetite",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you have nausea, vomiting, abdominal pain, bloating, or change in bowel habits?",
                        "hi": "क्या मितली, उल्टी, पेट दर्द, पेट फूलना, या मल त्याग की आदतों में बदलाव है?",
                        "gu": "શું તમને મિતલી, ઉલ્ટી, પેટનો દુખાવો, પેટ ફૂલવું અથવા પાખાનાની ટેવમાં ફેરફાર છે?",
                        "te": "మీకు వికారం, వాంతులు, కడుపు నొప్పి, బిగుసుకోవడం, లేదా మల విసర్జన అలవాటులో మార్పులు ఉన్నాయా?"
                    },
                    0.9, "yesno",
                    category="nausea",
                    symptom="nausea"
                ),
                make_question(
                    {
                        "en": "Have you experienced weight loss, weakness, or night sweats recently?",
                        "hi": "क्या हाल में वजन घटा है, कमजोरी या रात में पसीना आता है?",
                        "gu": "શું હમણાં વજન ઘટવું, કમજોરી કે રાત્રે પસીનો આવવો અનુભવ્યો છે?",
                        "te": "ఇటీవలి కాలంలో మీ బరువు తగ్గిందా, బలహీనంగా అనిపించిందా, లేదా రాత్రిళ్ళు చెమటలు పడుతున్నాయా?"
                    },
                    0.95, "yesno",
                    category="weight loss",
                    symptom="weight loss",
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "Do you have fever, cough, or any chronic illness like diabetes, liver disease, or thyroid disorder?",
                        "hi": "क्या बुखार, खांसी, या कोई पुरानी बीमारी (जैसे मधुमेह, लीवर रोग, थायरॉइड विकार) है?",
                        "gu": "શું તમને તાવ, ઊધરસ કે કોઈ લાંબી બીમારી (જેમ કે ડાયાબિટીસ, લિવર રોગ, થાયરોઇડની તકલીફ) છે?",
                        "te": "మీకు జ్వరం, దగ్గు, లేదా షుగర్, లివర్ వ్యాధి, థైరాయిడ్ లాంటి దీర్ఘకాలిక వ్యాధి ఏదైనా ఉందా?"
                    },
                    0.9, "yesno",
                    category="fever",
                    symptom="fever"
                ),
                make_question(
                    {
                        "en": "Are you under stress, feeling sad, anxious, or losing interest in your usual activities?",
                        "hi": "क्या आप तनाव में हैं, उदास/चिंतित महसूस करते हैं, या सामान्य गतिविधियों में रुचि घट गई है?",
                        "gu": "શું તમે તણાવમાં છો, ઉદાસ/ચિંતા અનુભવો છો, અથવા તમારી રોજબરોજની પ્રવૃત્તિઓમાં રસ ઓછો થયો છે?",
                        "te": "మీకు ఒత్తిడి, నిరాశ, ఆందోళనగా అనిపిస్తున్నదా, లేదా మీకు ఇష్టమైన పనుల్లో ఆసక్తి తగ్గిందా?"
                    },
                    0.9, "yesno",
                    category="stress",
                    symptom="stress"
                ),
                make_question(
                    {
                        "en": "Do you consume alcohol, smoke, or take any medications (painkillers, antibiotics, antidepressants)?",
                        "hi": "क्या आप शराब पीते हैं, धूम्रपान करते हैं, या कोई दवाएं (दर्द निवारक, एंटीबायोटिक, एंटीडिप्रेसेंट) लेते हैं?",
                        "gu": "શું તમે દારૂ પીવો છો, ધુમ્રપાન કરો છો, અથવા કોઈ દવાઓ (પેનકિલર, એન્ટિબાયોટિક, એન્ટીડિપ્રેસન્ટ) લો છો?",
                        "te": "మీరు ఆల్కహాల్ తాగుతారా, పొగ త్రాగుతారా, లేదా నొప్పి మందులు, యాంటీబయాటిక్స్, యాంటీడిప్రెసెంట్‌లు వంటి మందులు తీసుకుంటున్నారా?"
                    },
                    0.9, "yesno",
                    category="alcohol",
                    symptom="alcohol"
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["Loss of appetite_Less hungry"] = loss_of_appetite_less_hungry

        # 6) Anxiety_Nervousness_Panic attack
        anxiety_nervousness_panic_attack = HybridSymptomNode(
            "Anxiety_Nervousness_Panic attack",
            ["anxiety", "nervousness", "panic attack"],
            [
                make_question(
                    {
                        "en": "Can you describe what happens during these episodes: what do you feel in your body and mind? (e.g., palpitations, sweating, choking, fear of dying, trembling)",
                        "hi": "इन एपिसोड के दौरान क्या होता है— शरीर और मन में क्या महसूस होता है? (जैसे धड़कन तेज़ होना, पसीना, घुटन, मौत का डर, कांपना)",
                        "gu": "આ એપિસોડ દરમિયાન શું થાય છે તે વર્ણવો— શરીર અને મનમાં તમે શું અનુભવો છો? (જેમ કે ધડકન તેજ થવી, પસીનો, ઘૂંટણ લાગવું, મરણનો ભય, કંપારી)",
                        "te": "ఈ ఎపిసోడ్‌ల సమయంలో ఏమి జరుగుతుందో వివరించండి— మీ శరీరం మరియు మనసులో ఏమి అనిపిస్తుంది? (ఉదా: గుండె వేగంగా కొట్టుకోవడం, చెమటలు పట్టడం, ఉక్కిరిబిక్కిరి, చావిపోతాననే భయం, వణుకు)"
                    },
                    0.95, "yesno",
                    category="episode_phenomenology",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "How long do these panic attack episodes last and how often do they occur? Do they happen suddenly or are they triggered by stress, crowd, or specific situations?",
                        "hi": "ये आतंक का दौरा कितनी देर चलते हैं और कितनी बार होते हैं? क्या ये अचानक होते हैं या तनाव, भीड़ या विशेष स्थितियों से ट्रिगर होते हैं?",
                        "gu": "આ પેનિક એટેકના એપિસોડ કેટલા સમય સુધી ચાલે છે અને કેટલા વખત થાય છે? શું તે અચાનક થાય છે કે તણાવ, ભીડ અથવા ખાસ પરિસ્થિતિઓથી ટ્રિગર થાય છે?",
                        "te": "ఈ పానిక్ ఎటాక్ ఎపిసోడ్‌లు ఎంతసేపు సాగుతాయి, ఎంత తరచుగా వస్తుంటాయి? అవి ఒక్కసారిగా వచ్చేస్తాయా, లేక ఒత్తిడి, జనసమ్మర్థం లేదా కొన్ని ప్రత్యేక పరిస్థితులతో ట్రిగర్ అవుతాయా?"
                    },
                    0.9, "yesno",
                    category="duration: frequency_triggers",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you experience chest pain, shortness of breath, dizziness, or tingling sensations?",
                        "hi": "क्या आपको सीने में दर्द, सांस फूलना, चक्कर या झनझनाहट होती है?",
                        "gu": "શું તમને છાતીમાં દુખાવો, શ્વાસ ફૂલવો, ચક્કર અથવા ઝણઝણાટ અનુભવાય છે?",
                        "te": "మీకు ఛాతి నొప్పి, ఊపిరి తీసుకోవడంలో ఇబ్బంది, తల తిరగటం లేదా గరగర/సుడులు తిరిగినట్టు (టింగ్లింగ్) అనిపిస్తున్నాయా?"
                    },
                    0.9, "yesno",
                    category="chest pain, shortness of breath, dizziness, tingling sensations",
                    symptom="chest pain, shortness of breath, dizziness, tingling sensations",
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "Have you noticed weight loss, heat intolerance, excessive sweating, or tremors in hands?",
                        "hi": "क्या वजन घटना, गर्मी सहन न होना, अत्यधिक पसीना आना, या हाथों में कंपन देखा है?",
                        "gu": "શું તમે વજન ઘટવું, ગરમી સહન ન થવી, વધારે પસીનો આવવો અથવા હાથમાં કંપારી જોયી છે?",
                        "te": "మీ బరువు తగ్గడం, వేడి తట్టుకోలేకపోవడం, అధికంగా చెమటలు పట్టడం లేదా చేతులు వణకడం గమనించారా?"
                    },
                    0.9, "yesno",
                    category="weight loss, tremor",
                    symptom="weight loss",
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "Do you have a history of diabetes, thyroid disease, hypertension, or recent use of caffeine, alcohol, or stimulants?",
                        "hi": "क्या मधुमेह, थायरॉइड रोग, उच्च रक्तचाप का इतिहास है, या हाल में कैफीन, अल्कोहल या उत्तेजक दवाओं का उपयोग किया है?",
                        "gu": "શું તમને ડાયાબિટીસ, થાયરોઇડ રોગ, હાઈ બ્લડ પ્રેશરનો ઇતિહાસ છે, અથવા તાજેતરમાં કેફીન, આલ્કોહોલ અથવા સ્ટિમ્યુલન્ટ દવાઓ લીધી છે?",
                        "te": "మీకు షుగర్, థైరాయిడ్ వ్యాధి, హై బిపి చరిత్ర ఉందా, లేదా ఇటీవలి కాలంలో కాఫీన్, ఆల్కహాల్ లేదా స్టిమ్యులెంట్ మందులు ఎక్కువగా తీసుకున్నారా?"
                    },
                    0.85, "yesno",
                    category="diabetes, thyroid, high blood pressure, alcohol, caffeine",
                    symptom="diabetes, high blood pressure"
                ),
                make_question(
                    {
                        "en": "Do you worry excessively about health, performance, or daily activities, and find it difficult to control your anxiety?",
                        "hi": "क्या आप स्वास्थ्य, प्रदर्शन या दैनिक गतिविधियों को लेकर अत्यधिक चिंतित रहते हैं और चिंता को नियंत्रित करना कठिन लगता है?",
                        "gu": "શું તમે આરોગ્ય, કામકાજ અથવા રોજિંદા કાર્યો વિશે વધારે ચિંતા કરો છો અને ચિંતા પર કાબૂ રાખવો મુશ્કેલ લાગે છે?",
                        "te": "మీ ఆరోగ్యం, పనితీరు లేదా రోజువారీ పనుల గురించి అతిగా ఆందోళన పడుతూ, ఆ ఆందోళనను నియంత్రించడం కష్టంగా అనిపిస్తుందా?"
                    },
                    0.9, "yesno",
                    category="excessive_worry_control",
                    symptom=None
                ),
            ],
            weight=1.25
        )
        self.hybrid_symptom_nodes["Anxiety_Nervousness_Panic attack"] = anxiety_nervousness_panic_attack

        # 7) Leg pain_Foot pain
        leg_pain_foot_pain = HybridSymptomNode(
            "Leg pain_Foot pain",
            ["leg pain", "foot pain"],
            [
                make_question(
                    {
                        "en": "Where exactly is the pain located in muscles, joints, or bones? Is it one-sided or both sides?",
                        "hi": "टांग में दर्द ठीक कहाँ है— मांसपेशियों, जोड़ों या हड्डियों में? क्या यह एक तरफ है या दोनों तरफ?",
                        "gu": "ટાંગમાં દુખાવો ચોક્કસ ક્યાં છે— પેશીઓમાં, સાંધામાં કે હાડકાંમાં? શું તે એક બાજુ છે કે બન્ને બાજુ?",
                        "te": "పాదం లేదా కాలి నొప్పి కచ్చితంగా ఎక్కడ ఉంది— కండరాల్లోనా, సంధుల్లోనా, లేక ఎముకలలోనా? అది ఒక వైపు మాత్రమేనా లేదా రెండువైపులా?"
                    },
                    0.9, "yesno",
                    category="location: muscle pain",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "When did the leg pain start and how does it progress: sudden or gradual? Is it related to walking or rest?",
                        "hi": "टांग में दर्द कब शुरू हुआ और कैसे बढ़ता है— अचानक या धीरे-धीरे? क्या यह चलने या आराम से संबंधित है?",
                        "gu": "ટાંગમાં દુખાવો ક્યારે શરૂ થયો અને કેવી રીતે વધે છે— અચાનક કે ધીમે ધીમે? શું તે ચાલવા સાથે જોડાયેલો છે કે આરામમાં પણ રહે છે?",
                        "te": "కాలి నొప్పి ఎప్పుడు మొదలైంది, అది ఎలా పెరుగుతుంది— ఒక్కసారిగా వచ్చిందా లేదా నెమ్మదిగా పెరుగుతుందా? అది నడిచేప్పుడు ఎక్కువగా ఉంటుందా, లేక విశ్రాంతికీ సంబంధమా?"
                    },
                    0.9, "yesno",
                    category="duration: leg pain",
                    symptom="leg pain"
                ),
                make_question(
                    {
                        "en": "Do you notice swelling, redness, warmth, or tenderness over the affected area?",
                        "hi": "क्या प्रभावित स्थान पर सूजन, लालिमा, गर्माहट या दबाने पर दर्द है?",
                        "gu": "શું અસરગ્રસ્ત વિસ્તારમાં સૂજન, લાલાશ, ગરમાવો અથવા દબાવવા પર દુખાવો થાય છે?",
                        "te": "ప్రభావిత ప్రాంతంలో వాపు, ఎర్రదనం, వేడి లేదా నొక్కితే నొప్పి ఉందా?"
                    },
                    0.9, "yesno",
                    category="swelling",
                    symptom="swelling"
                ),
                make_question(
                    {
                        "en": "Do you feel tingling, numbness, burning, or weakness in your legs or feet?",
                        "hi": "क्या पैरों या पंजों में झनझनाहट, सुन्नपन, जलन या कमजोरी है?",
                        "gu": "શું તમારી ટાંગો અથવા પગમાં ઝણઝણાટ, સુન્નપણું, સળવળાટ કે કમજોરી છે?",
                        "te": "మీ కాళ్ళలో లేదా పాదాల్లో గుల్లుగా (టింగ్లింగ్), నిస్సారంగా, మంటగా లేదా బలహీనంగా అనిపిస్తున్నదా?"
                    },
                    0.9, "yesno",
                    category="tingling, numbness, leg weakness",
                    symptom="tingling, numbness, leg weakness"
                ),
                make_question(
                    {
                        "en": "Have you had any recent injury, long travel, prolonged standing, or immobilization?",
                        "hi": "क्या हाल में चोट लगी, लंबी यात्रा की, लंबे समय तक खड़े रहे, या गतिहीन रहे हैं?",
                        "gu": "શું તમે તાજેતરમાં કોઈ ઇજા, લાંબી મુસાફરી, લાંબા સમય સુધી ઊભા રહેવું અથવા વધારે સમય અચળ રહ્યા છો?",
                        "te": "ఇటీవలి కాలంలో మీకు ఎలాంటి గాయం, పొడవు ప్రయాణం, ఎక్కువ సేపు నిలబడటం లేదా సుదీర్ఘంగా కదలకుండా ఉండటం జరిగిందా?"
                    },
                    0.9, "yesno",
                    category="history: injury",
                    symptom="injury",
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "Do you have diabetes, gout, varicose veins, or take any medications like cholesterol-lowering drugs?",
                        "hi": "क्या आपको मधुमेह, गाउट, वैरिकाज़ नसें हैं, या कोलेस्ट्रॉल कम करने वाली दवाएं लेते हैं?",
                        "gu": "શું તમને ડાયાબિટીસ, ગાઉટ, વેરિકોઝ વેન્સ છે, અથવા કોલેસ્ટ્રોલ ઘટાડતી દવાઓ લો છો?",
                        "te": "మీకు షుగర్, గౌట్, వేరికోజ్ veins ఉన్నాయా, లేక కొలెస్ట్రాల్ తగ్గించే మందులు తీసుకుంటున్నారా?"
                    },
                    0.85, "yesno",
                    category="diabetes, medication",
                    symptom="diabetes"
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["Leg pain_Foot pain"] = leg_pain_foot_pain

        # 8) Nausea_Vomiting
        nausea_vomiting = HybridSymptomNode(
            "Nausea_Vomiting",
            ["nausea", "vomiting"],
            [
                make_question(
                    {
                        "en": "When did the nausea or vomiting start: sudden or gradual, and how many episodes happened till now?",
                        "hi": "मितली या उल्टी कब शुरू हुई— अचानक या धीरे-धीरे— और अब तक कितनी बार हुई है?",
                        "gu": "મિતલી અથવા ઉલ્ટી ક્યારે શરૂ થઈ— અચાનક કે ધીમે ધીમે— અને અત્યાર સુધી કેટલા એપિસોડ થયા છે?",
                        "te": "వికారం లేదా వాంతులు ఎప్పుడు మొదలయ్యాయి— ఒక్కసారిగా వచ్చాయా లేదా నెమ్మదిగా ప్రారంభమయ్యాయా— ఇప్పటివరకు ఎన్ని సార్లు వచ్చాయి?"
                    },
                    0.9, "yesno",
                    category="duration: nausea, vomiting",
                    symptom="nausea, vomiting"
                ),
                make_question(
                    {
                        "en": "Is there any relation to food intake: before taking food, after taking food, or irrespective of food intake?",
                        "hi": "क्या इसका भोजन से संबंध है: खाने से पहले, खाने के बाद, या भोजन से बिना संबंध?",
                        "gu": "શું આનો ભોજન સાથે કોઈ સંબંધ છે: ખાવા પહેલાં, ખાવા પછી, કે ભોજનથી અલગ પણ થાય છે?",
                        "te": "ఇది భోజనంతో సంబంధం ఉందా— తినే ముందు, తిన్న తర్వాతా, లేక భోజనానికి సంబంధం లేకుండా కూడా వస్తుందా?"
                    },
                    0.9, "yesno",
                    category="diet: vomiting",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you have associated abdominal pain, distension, constipation, diarrhea, or headache?",
                        "hi": "क्या पेट दर्द, पेट फूलना, कब्ज, दस्त या सिरदर्द जैसे लक्षण साथ हैं?",
                        "gu": "શું સાથે પેટનો દુખાવો, પેટ ફૂલવું, કબજિયાત, દસ્ત અથવા માથાનો દુખાવો છે?",
                        "te": "దీనితో పాటు మీకు కడుపు నొప్పి, పొత్తికడుపు బిగుసుకోవడం, మలబద్ధకం, డయ్యేరియా లేదా తలనొప్పి ఉన్నాయా?"
                    },
                    0.9, "yesno",
                    category="stomach pain, diarrhea",
                    symptom="stomach pain, diarrhea"
                ),
                make_question(
                    {
                        "en": "Are there other symptoms like dizziness, visual blurring, fever, or decreased urine output?",
                        "hi": "क्या चक्कर, धुंधला दिखना, बुखार या पेशाब में कमी जैसे अन्य लक्षण हैं?",
                        "gu": "શું ચક્કર, ધૂંધળી નજર, તાવ અથવા મૂત્રમાં ઘટાડો જેવા અન્ય લક્ષણો છે?",
                        "te": "మరేమైనా లక్షణాలున్నాయా, ఉదా: తల తిరగటం, చూపు మసకబారడం, జ్వరం, లేదా మూత్రం తక్కువగా రావడం?"
                    },
                    0.9, "yesno",
                    category="dizziness, blurred vision, fever",
                    symptom="dizziness, blurred vision, fever"
                ),
                make_question(
                    {
                        "en": "Do you have diabetes, kidney/liver disease, or are you taking any regular medications?",
                        "hi": "क्या आपको मधुमेह, किडनी/लिवर रोग है या आप कोई नियमित दवाएं लेते हैं?",
                        "gu": "શું તમને ડાયાબિટીસ, કિડની/લિવરનો રોગ છે અથવા તમે કોઈ નિયમિત દવાઓ લો છો?",
                        "te": "మీకు షుగర్, కిడ్నీ/లివర్ వ్యాధి ఉందా లేదా మీరు ఏవైనా మందులు నిత్యం తీసుకుంటున్నారా?"
                    },
                    0.85, "yesno",
                    category="diabetes, medication",
                    symptom=None
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["Nausea_Vomiting"] = nausea_vomiting

        # 9) Chest pain_ Heart Pain
        chest_pain_heart_pain = HybridSymptomNode(
            "Chest pain_ Heart Pain",
            ["chest pain", "heart pain"],
            [
                make_question(
                    {
                        "en": "Can you describe the chest pain: is it heaviness on chest, pressure on chest, stabbing, burning?",
                        "hi": "सीने के दर्द का वर्णन करें: क्या यह सीने पर भारीपन/दबाव, चुभन या जलन जैसा है?",
                        "gu": "છાતીનો દુખાવો કેવો છે તે જણાવો: શું તે છાતી પર ભાર/દબાણ જેવો, ચભચભતો કે સળવળાટ જેવો છે?",
                        "te": "ఛాతీ నొప్పి ఎలా ఉందో చెప్పండి— అది ఛాతీపై బరువుగా, ఒత్తిడిగా, గుచ్చినట్టు లేదా మంటగా ఉందా?"
                    },
                    0.95, "yesno",
                    category="pain_character_chest",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Does the pain radiate to any other area: like the left arm, jaw, shoulder, neck, or back?",
                        "hi": "क्या दर्द बाएँ हाथ, जबड़े, कंधे, गर्दन या पीठ तक फैलता है?",
                        "gu": "શું દુખાવો ડાબા હાથ, જડબા, ખભા, ગળા અથવા પીઠ સુધી ફેલાય છે?",
                        "te": "ఈ నొప్పి ఎడమ చేయి, దవడ, భుజం, మెడ లేదా వెన్ను వరకు వ్యాపిస్తుందా?"
                    },
                    0.95, "yesno",
                    category="location: chest pain",
                    symptom=None,
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "When did the pain start, how long does it last, and is it brought on by exertion, stress, or rest?",
                        "hi": "दर्द कब शुरू हुआ, कितनी देर रहता है, और क्या यह मेहनत, तनाव या आराम में आता है?",
                        "gu": "દુખાવો ક્યારે શરૂ થયો, કેટલો સમય રહે છે, અને શું તે મહેનત, તણાવ અથવા આરામ કરતી વખતે થાય છે?",
                        "te": "ఈ నొప్పి ఎప్పుడు మొదలైంది, ఎంతసేపు ఉంటుంది, శ్రమ, ఒత్తిడి సమయంలోనా లేక విశ్రాంతిలోనూ వస్తుందా?"
                    },
                    0.9, "yesno",
                    category="duration: chest pain",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you have other symptoms like sweating, breathlessness, nausea, or fainting?",
                        "hi": "क्या पसीना आना, सांस फूलना, मितली या बेहोशी जैसे अन्य लक्षण हैं?",
                        "gu": "શું તમને વધારે પસીનો આવવો, શ્વાસ ફૂલવો, મિતલી કે બેભાન થવા જેવા અન્ય લક્ષણો છે?",
                        "te": "మీకు చెమటలు పట్టడం, ఊపిరి బిగుసుకోవడం, వికారం లేదా మూర్చల వంటి మరిన్ని లక్షణాలున్నాయా?"
                    },
                    0.95, "yesno",
                    category="sweating, shortness of breath, nausea",
                    symptom="sweating, shortness of breath, nausea",
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "Does the pain change with body position, deep breathing, or movement?",
                        "hi": "क्या शरीर की स्थिति, गहरी सांस या हिलने-डुलने से दर्द बदलता है?",
                        "gu": "શું શરીરની સ્થિતિ બદલવાથી, ઊંડો શ્વાસ લેવા અથવા હિલચાલથી દુખાવો બદલાય છે?",
                        "te": "శరీరాన్ని తిప్పినప్పుడు, లోతుగా శ్వాస తీసుకున్నప్పుడు లేదా కదలినప్పుడు నొప్పి మారుతుందా?"
                    },
                    0.9, "yesno",
                    category="positional_pleuritic_msk_clues",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you have hypertension, diabetes, heart disease, high cholesterol, smoking habit, or any family history of heart disease?",
                        "hi": "क्या उच्च रक्तचाप, मधुमेह, हृदय रोग, उच्च कोलेस्ट्रॉल, धूम्रपान या हृदय रोग का पारिवारिक इतिहास है?",
                        "gu": "શું તમને હાઈ બ્લડ પ્રેશર, ડાયાબિટીસ, હૃદય રોગ, વધુ કોલેસ્ટ્રોલ, ધુમ્રપાનની આદત અથવા હૃદયરોગનો પારિવારિક ઇતિહાસ છે?",
                        "te": "మీకు హై బిపి, షుగర్, గుండె వ్యాధి, అధిక కొలెస్ట్రాల్, పొగ త్రాగే అలవాటు లేదా కుటుంబంలో గుండె వ్యాధి చరిత్ర ఉందా?"
                    },
                    0.9, "yesno",
                    category="smoking, diabetes",
                    symptom=None
                ),
            ],
            weight=1.3
        )
        self.hybrid_symptom_nodes["Chest pain_ Heart Pain"] = chest_pain_heart_pain

        # 10) cough_acid
        acidity_cough = HybridSymptomNode(
            "acidity_cough",
            ["acidity", "cough"],
            [
                make_question(
                    {
                        "en": "When did the cough start, and is it dry or productive? Does the cough occur more at night or after meals?",
                        "hi": "खांसी कब शुरू हुई, और क्या यह सूखी है या बलगम वाली? क्या खांसी रात में या खाने के बाद अधिक होती है?",
                        "gu": "ઉધરસ ક્યારે શરૂ થયો, અને શું તે સૂકો છે કે કફવાળો? શું ઉધરસ રાત્રે અથવા ભોજન પછી વધુ થાય છે?",
                        "te": "దగ్గు ఎప్పుడు ప్రారంభమైంది, అది పొడి దగ్గునా లేదా కఫం వస్తూ దగ్గునా? దగ్గు రాత్రిపూటా లేదా భోజనం తర్వాత ఎక్కువగా వస్తుందా?"
                    },
                    0.9, "yesno",
                    category="duration: cough",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you experience burning in the chest, a sour taste in the mouth, or regurgitation—especially after eating or when lying down?",
                        "hi": "क्या सीने में जलन, मुंह में खट्टा स्वाद या खाना वापस आने का अनुभव होता है—खासकर खाने के बाद या लेटने पर?",
                        "gu": "શું તમને છાતીમાં સળવળાટ, મોઢામાં ખાટો સ્વાદ કે ખોરાક ઉપર ચડી આવવો અનુભવાય છે—ખાસ કરીને ખાવા પછી અથવા લેટતાં?",
                        "te": "భోజనం చేసిన తర్వాత లేదా పరవశంగా పడుకున్నప్పుడు ఛాతీలో మంట, నోటిలో పులుపు, లేదా ఆహారం పైకి రావడం అనిపిస్తుందా?"
                    },
                    0.9, "yesno",
                    category="GERD symptoms: heartburn/regurgitation",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Does the cough worsen when you lie flat, talk, or after large meals?",
                        "hi": "क्या सीधा लेटने, बात करने या ज़्यादा खाना खाने के बाद खांसी बढ़ जाती है?",
                        "gu": "શું સીધા લેટવાથી, વધુ વાતચીતથી અથવા વધારે ભોજન બાદ ઉધરસ વધી જાય છે?",
                        "te": "మీరు సూటిగా పడుకున్నప్పుడు, ఎక్కువ మాట్లాడినప్పుడు లేదా ఎక్కువగా తిన్న తర్వాత దగ్గు ఎక్కువ అవుతుందా?"
                    },
                    0.9, "yesno",
                    category="triggers: supine/talking/large meals",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you have hoarseness of voice, frequent throat clearing, or a sensation of a lump in the throat?",
                        "hi": "क्या आवाज़ बैठना, बार-बार गला साफ करना या गले में गाँठ जैसा महसूस होना होता है?",
                        "gu": "શું તમને અવાજ બેસી જવો, વારંવાર ગળો સાફ કરવાની ઇચ્છા કે ગળામાં ગાંઠ જેવી લાગણી થાય છે?",
                        "te": "మీ స్వరం ముడుపుగా మారడం, తరచూ గొంతు ఖరాఖరా చేసుకోవాలనిపించడం, లేదా గొంతులో ముద్దలాగా అనిపించడం ఉందా?"
                    },
                    0.9, "yesno",
                    category="laryngopharyngeal reflux signs",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Have you noticed breathlessness, chest pain, or wheezing?",
                        "hi": "क्या सांस फूलना, सीने में दर्द या घरघराहट होती है?",
                        "gu": "શું તમને શ્વાસ ફૂલે છે, છાતીમાં દુખાવો કે સીટી જેવી અવાજ સાથે શ્વાસ (ઘરઘરાહટ) આવે છે?",
                        "te": "మీకు ఊపిరి బిగుసుకుపోవడం, ఛాతీ నొప్పి లేదా వీసం (వీజింగ్) ఉందా?"
                    },
                    0.95, "yesno",
                    category="shortness of breath,chest pain,wheezing",
                    symptom="shortness of breath, chest pain, wheezing",
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "Are you on any long-term medications for blood pressure or painkillers (e.g., ACE inhibitors, NSAIDs)?",
                        "hi": "क्या आप लंबे समय से ब्लड प्रेशर की दवाएं या दर्द निवारक (जैसे ACE इन्हिबिटर, NSAIDs) ले रहे हैं?",
                        "gu": "શું તમે લાંબા સમયથી બ્લડ પ્રેશરની દવાઓ અથવા પેનકિલર (જેમ કે ACE ઇનહિબિટર, NSAIDs) લઈ રહ્યા છો?",
                        "te": "మీరు చాలా రోజులుగా బిపి మందులు లేదా నొప్పి మందులు (ఉదా: ACE ఇన్హిబిటర్లు, NSAIDs) తీసుకుంటున్నారా?"
                    },
                    0.85, "yesno",
                    category="medication",
                    symptom=None
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["acidity_cough"] = acidity_cough

        # 11) acidity_stomachpain
        acidity_stomachpain = HybridSymptomNode(
            "acidity_stomachpain",
            ["acidity", "stomach_pain"],
            [
                make_question(
                    {
                        "en": "Where exactly is the stomach pain—upper abdomen (epigastric), central, or does it radiate to the back?",
                        "hi": "पेट में दर्द ठीक कहाँ है— ऊपरी पेट (एपिगैस्ट्रिक), बीच में, या क्या यह पीठ तक फैलता है?",
                        "gu": "પેટમાં દુખાવો ચોક્કસ ક્યાં છે— ઉપરલા પેટમાં (એપિગાસ્ટ્રિક), મધ્યમાં, કે શું તે પીઠ સુધી ફેલાય છે?",
                        "te": "కడుపు నొప్పి కచ్చితంగా ఎక్కడ ఉంది— పై కడుపు (ఎపిగాస్ట్రిక్ ప్రాంతం), మధ్య భాగంలోనా, లేక వెన్ను వైపు పాకుతున్నదా?"
                    },
                    0.9, "yesno",
                    category="location: epigastric/central/radiation",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "When does the stomach pain occur—before meals, after meals, or at night? Does food or taking antacids relieve it?",
                        "hi": "पेट दर्द कब होता है— खाने से पहले, खाने के बाद, या रात में? क्या खाना या एंटासिड लेने से दर्द में राहत मिलती है?",
                        "gu": "પેટનો દુખાવો ક્યારે થાય છે— ખાવા પહેલાં, ખાવા પછી કે રાત્રે? શું ખાવાથી અથવા એન્ટાસિડ લેવાથી દુખાવો ઓછો થાય છે?",
                        "te": "ఈ కడుపు నొప్పి ఎప్పుడు వస్తుంది— తినే ముందు, తిన్న తర్వాతా, లేక రాత్రిళ్ళు? ఆహారం తినడం లేదా యాంటాసిడ్ తీసుకోవడం వల్ల నొప్పి తగ్గుతుందా?"
                    },
                    0.9, "yesno",
                    category="timing/response: pre/post-meal, nocturnal; antacid relief",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you have nausea, bloating, or a feeling of fullness early during small meals?",
                        "hi": "क्या मितली, पेट फूलना, या थोड़ा खाने पर ही पेट भर जाने जैसा महसूस होता है?",
                        "gu": "શું તમને મિતલી, પેટ ફૂલવું અથવા થોડું ખાવાથી જ પેટ ભરાઈ ગયું હોય તેવી લાગણી થાય છે?",
                        "te": "మీకు వికారం, కడుపు బిగుసుకోవడం, లేదా కొద్దిగా తిన్న వెంటనే నిండిపోయినట్టు అనిపిస్తుందా?"
                    },
                    0.9, "yesno",
                    category="associated dyspepsia symptoms",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Have you noticed black-colored stools, vomiting of blood, or unintended weight loss?",
                        "hi": "क्या काला मल आया है, खून की उल्टी हुई है, या बिना कारण वजन घटा है?",
                        "gu": "શું તમને કાળો મલ આવ્યો છે, લોહીવાળી ઉલ્ટી થઈ છે અથવા વિનાકારણ વજન ઘટ્યું છે?",
                        "te": "మీకు నల్లటి మలం, రక్తం కలిసిన వాంతులు లేదా కారణం లేకుండా బరువు తగ్గటం గమనించారా?"
                    },
                    0.95, "yesno",
                    category="red flags: melena/hematemesis/weight loss",
                    symptom=None,
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "Do you consume alcohol, smoke, or frequently take painkiller medicines?",
                        "hi": "क्या आप शराब पीते हैं, धूम्रपान करते हैं, या अक्सर दर्द निवारक दवाएं लेते हैं?",
                        "gu": "શું તમે દારૂ પીવો છો, ધુમ્રપાન કરો છો અથવા વારંવાર પેનકિલર દવાઓ લો છો?",
                        "te": "మీరు ఆల్కహాల్ తాగుతారా, పొగ త్రాగుతారా, లేదా నొప్పి మందులు తరచూ తీసుకుంటున్నారా?"
                    },
                    0.85, "yesno",
                    category="risk habits: alcohol/smoking/NSAIDs",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Have you ever been tested or treated for stomach ulcers, or gallbladder or pancreatic problems before?",
                        "hi": "क्या पहले कभी पेट के अल्सर, पित्ताशय या अग्न्याशय की समस्याओं की जाँच/इलाज हुआ है?",
                        "gu": "શું તમને પહેલાં ક્યારેય પેટના અલ્સર, પિત્તાશય અથવા અગ્ન્યાશયની તકલીફ માટે પરીક્ષણ/ઉપચાર થયો છે?",
                        "te": "మీరు ఇంతకుముందు కడుపు పుండ్లు, పిత్తాశయ (గాల్ బ్లాడర్) లేదా ప్యాంక్రియాస్ సమస్యల కోసం పరీక్షలు/చికిత్స చేయించుకున్నారా?"
                    },
                    0.85, "yesno",
                    category="prior GI history: ulcers/gallbladder/pancreas",
                    symptom=None
                ),
            ],
            weight=1.25
        )
        self.hybrid_symptom_nodes["acidity_stomachpain"] = acidity_stomachpain

        # 1) Acne_Stomach Pain
        acne_stomach_pain = HybridSymptomNode(
            "Acne_Stomach Pain",
            ["acne", "stomach pain"],
            [
                make_question(
                    {
                        "en": "When did the acne start? Is it recent or long-standing?",
                        "hi": "मुहांसे कब से हैं? क्या यह हाल ही में शुरू हुए हैं या लंबे समय से हैं?",
                        "gu": "મોં પરનાં દાણા (એકને) ક્યારેથી છે? શું તે તાજેતરમાં શરૂ થયા છે કે લાંબા સમયથી છે?",
                        "te": "మొటిమలు (ఐక్ని) ఎప్పటి నుంచి ఉన్నాయి? ఇవి కొత్తగా వచ్చాయా లేదా చాలాకాలంగా ఉన్నాయా?"
                    },
                    0.9, "yesno",
                    category="duration: acne",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Are you taking any medicines such as steroids, protein supplements, or isotretinoin tablets?",
                        "hi": "क्या आप स्टेरॉयड, प्रोटीन सप्लीमेंट या आइसोट्रेटिनॉइन टैबलेट जैसी दवाएं ले रहे हैं?",
                        "gu": "શું તમે સ્ટેરોઇડ, પ્રોટીન સપ્લિમેન્ટ્સ અથવા આઇસોટ્રેટિનોઇન જેવી ગોળીઓ લઈ રહ્યા છો?",
                        "te": "మీరు స్టెరాయిడ్లు, ప్రోటీన్ సప్లిమెంట్లు లేదా ఐసోట్రెటినాయిన్ వంటి మందులు తీసుకుంటున్నారా?"
                    },
                    0.9, "yesno",
                    category="medications: protein",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Can you describe your stomach pain—where is it located; does it increase after meals, or is it relieved by antacids?",
                        "hi": "अपने पेट दर्द का वर्णन करें— दर्द कहाँ होता है? क्या यह खाने के बाद बढ़ता है, या एंटासिड लेने से राहत मिलती है?",
                        "gu": "તમારા પેટના દુખાવાનું વર્ણન કરો— દુખાવો ક્યાં થાય છે? શું તે ભોજન પછી વધે છે કે એન્ટાસિડ લેવાથી રાહત મળે છે?",
                        "te": "మీ కడుపు నొప్పి ఎలా ఉంటుందో వివరించండి— అది ఎక్కడ ఉంది? భోజనం చేసిన తర్వాత పెరుగుతుందా, లేదా యాంటాసిడ్ తీసుకుంటే తగ్గిపోతుందా?"
                    },
                    0.9, "yesno",
                    category="location: stomach pain",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you have symptoms like bloating, nausea, or irregular bowel habits?",
                        "hi": "क्या आपको पेट फूलना, मितली या मल त्याग की आदतों में अनियमितता जैसे लक्षण हैं?",
                        "gu": "શું તમને પેટ ફૂલવું, મિતલી કે પાખાનાની ટેવમાં અનિયમિતતા જેવા લક્ષણો છે?",
                        "te": "మీకు కడుపు బిగుసుకోవడం, వికారం లేదా మల విసర్జన అలవాటులో మార్పులు ఉన్నాయా?"
                    },
                    0.9, "yesno",
                    category="bloating, nausea, indigestion",
                    symptom="bloating"
                ),
                make_question(
                    {
                        "en": "Do you consume spicy foods, caffeine, or oily foods frequently?",
                        "hi": "क्या आप अक्सर मसालेदार, कैफीनयुक्त या तैलीय भोजन लेते हैं?",
                        "gu": "શું તમે વારંવાર મસાલેદાર, કેફીનવાળા અથવા તેલિયાં ખોરાક લો છો?",
                        "te": "మీరు తరచుగా మసాలా, కాఫీ/టీ (కెఫీన్), లేదా ఆయిల్ ఎక్కువగా ఉన్న పదార్థాలు తింటున్నారా?"
                    },
                    0.85, "yesno",
                    category="diet: spicy",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you experience stress and poor sleep?",
                        "hi": "क्या आपको तनाव रहता है और नींद ठीक से नहीं आती?",
                        "gu": "શું તમને તણાવ રહે છે અને ઊંઘ સારી આવતી નથી?",
                        "te": "మీకు ఒత్తిడి ఎక్కువగా ఉంటుందా, నిద్ర బాగా పడట్లేదా?"
                    },
                    0.85, "yesno",
                    category="stress",
                    symptom="stress"
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["Acne_Stomach Pain"] = acne_stomach_pain

        # 2) Allergy_blister
        allergy_blister = HybridSymptomNode(
            "Allergy_blister",
            ["allergy", "blister"],
            [
                make_question(
                    {
                        "en": "When did the blisters first appear?",
                        "hi": "छाले पहली बार कब दिखाई दिए?",
                        "gu": "છાલા સૌથી પહેલા ક્યારે દેખાયા?",
                        "te": "ఈ బుడగలు (బ్లిస్టర్లు) మొదట ఎప్పుడూ కనిపించాయి?"
                    },
                    0.9, "yesno",
                    category="duration: blisters",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Did the blisters appear after exposure to any new food, drug, cosmetic, or chemical?",
                        "hi": "क्या नए भोजन, दवा, कॉस्मेटिक या रसायन के संपर्क के बाद छाले हुए?",
                        "gu": "શું નવા ખોરાક, દવા, કોસ્મેટિક અથવા રસાયણના સંપર્ક પછી છાલા થયા?",
                        "te": "ఏదైనా కొత్త ఆహారం, మందు, కాస్మెట్టిక్ లేదా కెమికల్‌కి ఎక్స్‌పోజర్ అయిన తర్వాత ఈ బుడగలు వచ్చాయా?"
                    },
                    0.9, "yesno",
                    category="cause: blister",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Where did the blisters begin, and are they localized or widespread?",
                        "hi": "छाले कहाँ से शुरू हुए, और क्या वे सीमित हैं या पूरे शरीर में फैले हैं?",
                        "gu": "છાલા ક્યા ભાગથી શરૂ થયા, અને શું તેઓ માત્ર ત્યાં સુધી સીમિત છે કે આખા શરીરમાં ફેલાયા છે?",
                        "te": "ఈ బుడగలు ముందుగా ఎక్కడ మొదలయ్యాయి, అవి ఒక్కచోటే ఉన్నాయా లేదా శరీరమంతా వ్యాపిస్తున్నాయా?"
                    },
                    0.9, "yesno",
                    category="location: blister",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Are the blisters itchy, painful, or burning?",
                        "hi": "क्या छालों में खुजली, दर्द या जलन होती है?",
                        "gu": "શું છાલાઓમાં ખુજલી, દુખાવો કે સળવળાટ થાય છે?",
                        "te": "ఈ బుడగలు గట్టిగా దురదగా, నొప్పిగా లేదా మంటగా ఉన్నాయా?"
                    },
                    0.9, "yesno",
                    category="itchy: blister",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Did you have fever, malaise, or a rash before the blisters appeared?",
                        "hi": "क्या छाले आने से पहले बुखार, थकान या चकत्ते हुए थे?",
                        "gu": "છાલા આવતાં પહેલા શું તમને તાવ, થાક લાગવો અથવા શરીર પર ચકામા (રેશ) હતા?",
                        "te": "బుడగలు వచ్చేముందు మీకు జ్వరం, బలహీనత లేదా చర్మంపై దద్దుర్లు వచ్చాయా?"
                    },
                    0.9, "yesno",
                    category="fever",
                    symptom="fever",
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "Have you recently taken any medicines such as antibiotics, painkillers, or any new medicines?",
                        "hi": "क्या आपने हाल में एंटीबायोटिक्स, दर्द निवारक या कोई नई दवाएं ली हैं?",
                        "gu": "શું તમે હમણાં કોઈ એન્ટિબાયોટિક, પેનકિલર અથવા નવી દવાઓ લીધી છે?",
                        "te": "ఇటీవల మీరు యాంటీబయోటిక్స్, నొప్పి మందులు లేదా మరే కొత్త మందులు తీసుకున్నారా?"
                    },
                    0.9, "yesno",
                    category="medication: antibiotics",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you have any history of asthma, eczema or autoimmune disease? Any similar episodes in the past?",
                        "hi": "क्या दमा, एक्ज़िमा या ऑटोइम्यून रोग का इतिहास है? क्या पहले भी ऐसे एपिसोड हुए हैं?",
                        "gu": "શું તમને દમા, એક્ઝિમા અથવા કોઈ ઓટોઇમ્યૂન રોગનો ઇતિહાસ છે? શું પહેલાં પણ આવાં એપિસોડ થયા છે?",
                        "te": "మీకు ఆస్తమా, ఎగ్జిమా లేదా ఆటోఇమ్యూన్ వ్యాధుల చరిత్ర ఉందా? ఇంతకుముందు కూడా ఇలాంటి ఎపిసోడ్‌లున్నాయా?"
                    },
                    0.85, "yesno",
                    category="history: atopy/autoimmune/recurrence",
                    symptom=None
                ),
            ],
            weight=1.25
        )
        self.hybrid_symptom_nodes["Allergy_blister"] = allergy_blister

        # 3) Fever_Handpain_Swelling
        fever_handpain_swelling = HybridSymptomNode(
            "Fever_Handpain_Swelling",
            ["fever", "hand pain", "swelling"],
            [
                make_question(
                    {
                        "en": "When did the fever and hand swelling start—and which came first?",
                        "hi": "बुखार और हाथ में सूजन कब शुरू हुई— और पहले क्या शुरू हुआ?",
                        "gu": "તાવ અને હાથમાં સૂજન ક્યારે શરૂ થયું— અને પહેલા શું આવ્યું?",
                        "te": "జ్వరం మరియు చెయ్యి వాపు ఎప్పుడు ప్రారంభమయ్యాయి— ముందుగా ఏది మొదలైంది?"
                    },
                    0.9, "yesno",
                    category="instance: fever vs swelling",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Is the pain and swelling limited to one hand or one joint, or does it involve multiple small joints of both hands?",
                        "hi": "क्या दर्द और सूजन एक हाथ/एक जोड़ तक सीमित है, या दोनों हाथों के कई छोटे जोड़ों को प्रभावित करती है?",
                        "gu": "શું દુખાવો અને સૂજન એક હાથ/એક સાંધા સુધી સીમિત છે કે બન્ને હાથના ઘણા નાના સાંધાઓને અસર કરે છે?",
                        "te": "నొప్పి, వాపు ఒక్క చెయ్యి/ఒక్క సంధి వరకు మాత్రమే ఉన్నాయా, లేక రెండు చేతుల చిన్న చిన్న సంధుల్లో కూడా ఉన్నాయా?"
                    },
                    0.9, "yesno",
                    category="location: hand pain",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you notice stiffness of the fingers in the morning?",
                        "hi": "क्या सुबह उंगलियों में जकड़न महसूस होती है?",
                        "gu": "શું તમને સવારે આંગળીઓમાં જકડન અનુભવાય છે?",
                        "te": "మీకు ఉదయాన్నే వేలు గట్టిపడినట్టు, కదలకుండా ఉన్నట్టు అనిపిస్తుందా?"
                    },
                    0.9, "yesno",
                    category="morning stiffness",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Is there redness or tenderness over the joint? Does the hand pain reduce with rest or with movement?",
                        "hi": "क्या जोड़ पर लालिमा या दबाने पर दर्द है? क्या आराम करने से या चलाने से हाथ के दर्द में कमी आती है?",
                        "gu": "શું સાંધા પર લાલાશ કે દબાવવાથી દુખાવો થાય છે? હાથને આરામ આપવાથી કે હલાવવાથી દુખાવો ઘટે છે?",
                        "te": "సంధి మీద ఎర్రదనం లేదా నొక్కితే నొప్పి ఉందా? చేతిని కదపకుండా విశ్రాంతి ఇస్తే నొప్పి తగ్గుతుందా, లేక కదిలిస్తే తగ్గుతుందా?"
                    },
                    0.9, "yesno",
                    category="inflammation/tenderness; pain_variation_rest/movement",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Have you recently had a sore throat, skin infection, or insect bite before the joint symptoms began?",
                        "hi": "क्या जोड़ों के लक्षण शुरू होने से पहले गले में खराश, त्वचा संक्रमण या कीड़े के काटने का अनुभव हुआ था?",
                        "gu": "શું સાંધાના લક્ષણો શરૂ થાય તે પહેલાં તમને ગળામાં દુખાવો, ત્વચાનો ઇન્ફેક્શન અથવા જીવાતનું ડંખ થયું હતું?",
                        "te": "సంధి సమస్యలు మొదలయ్యే ముందు మీకు గొంతు నొప్పి, చర్మ ఇన్ఫెక్షన్ లేదా పురుగు కాటు జరిగినదా?"
                    },
                    0.9, "yesno",
                    category="sore throat",
                    symptom="sore throat"
                ),
            ],
            weight=1.25
        )
        self.hybrid_symptom_nodes["Fever_Handpain_Swelling"] = fever_handpain_swelling

        # 4) Headache_Highblood pressure_legbleeding
        headache_hbp_legbleeding = HybridSymptomNode(
            "Headache_Highblood pressure_legbleeding",
            ["headache", "high blood pressure", "leg bleeding"],
            [
                make_question(
                    {
                        "en": "When did the headache start—was it sudden or persistent for a long time? Is it severe or dull?",
                        "hi": "सिरदर्द कब शुरू हुआ— क्या यह अचानक शुरू हुआ या लंबे समय से बना हुआ है? क्या यह तेज़ है या हल्का?",
                        "gu": "માથાનો દુખાવો ક્યારે શરૂ થયો— શું તે અચાનક શરૂ થયો કે લાંબા સમયથી છે? શું તે ખૂબ તીવ્ર છે કે હળવો?",
                        "te": "తలనొప్పి ఎప్పుడు ప్రారంభమైంది— అది ఒక్కసారిగా మొదలయ్యిందా లేదా చాలాకాలంగా కొనసాగుతుందా? అది బలంగా నొప్పిదా లేదా తేలికపాటి నొప్పా?"
                    },
                    0.9, "yesno",
                    category="duration: headache",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Were you taking any medicines to control blood pressure regularly?",
                        "hi": "क्या आप नियमित रूप से ब्लड प्रेशर नियंत्रित करने की दवाएं ले रहे थे?",
                        "gu": "શું તમે નિયમિત રીતે બ્લડ પ્રેશર નિયંત્રિત કરવાની દવાઓ લઇ રહ્યા હતા?",
                        "te": "మీరు బ్లడ్ ప్రెజర్ నియంత్రించడానికి మందులు క్రమంగా తీసుకుంటూ ఉన్నారా?"
                    },
                    0.9, "yesno",
                    category="medication: high blood pressure",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Have you noticed nosebleeds, bruising, or bleeding from other sites apart from the leg bleeding?",
                        "hi": "क्या पैर से रक्तस्राव के अलावा नकसीर, खरोंच/नील पड़ना, या अन्य जगह से खून आना देखा है?",
                        "gu": "પગમાંથી રક્તસ્ત્રાવ સિવાય શું તમને નાકમાંથી લોહી આવવું, શરીર પર સરળે વાગતાં નીલ પડી જવી અથવા અન્ય ક્યાંયથી લોહી આવવું દેખાયું છે?",
                        "te": "కాలు నుంచి రక్తస్రావం కాకుండా మీకు ముక్కు నుంచి రక్తం కారడం, శరీరంపై నీలికలు/గాయాలు సులభంగా రావడం, లేదా మరే చోట రక్తస్రావం గమనించారా?"
                    },
                    0.95, "yesno",
                    category="bleeding : nose",
                    symptom=None,
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "Was the leg bleeding due to an injury/trauma, or did it start spontaneously?",
                        "hi": "क्या पैर से खून चोट/ट्रॉमा के कारण हुआ, या स्वतः शुरू हो गया?",
                        "gu": "પગમાંથી રક્તસ્ત્રાવ કોઈ ઈજા/ટ્રોમા કારણે થયો હતો કે પોતે જ શરૂ થયો?",
                        "te": "కాలు నుంచి రక్తం రావడం గాయం/ప్రమాదం వల్ల జరిగిందా, లేదా ఎలాంటి కారణం లేకుండానే మొదలైందా?"
                    },
                    0.9, "yesno",
                    category="cause: injury",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you have swelling, pain, or discoloration of the leg? Any signs of varicose veins or ulceration?",
                        "hi": "क्या पैर में सूजन, दर्द या रंग बदलना है? वैरिकाज़ नसें या अल्सर के कोई लक्षण हैं?",
                        "gu": "શું તમારા પગમાં સૂજન, દુખાવો કે રંગ બદલાવ છે? વેરિકોઝ વેન્સ અથવા ઘા/અલ્સર જેવા લક્ષણો છે?",
                        "te": "కాళ్ళలో వాపు, నొప్పి లేదా రంగు మారటం ఉందా? బయట చిక్కుకొన్న రక్తనాళాలు (వరికోజ్ veins) లేదా గాయాలు/పోకలు వంటి లక్షణాలున్నాయా?"
                    },
                    0.9, "yesno",
                    category="pain: leg",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Are you on any medications such as aspirin, blood thinners, or heart-related medicines?",
                        "hi": "क्या आप एस्पिरिन, ब्लड थिनर या हृदय संबंधी दवाएं ले रहे हैं?",
                        "gu": "શું તમે એસ્પિરિન, બ્લડ થિનર અથવા હૃદય સંબંધિત દવાઓ લો છો?",
                        "te": "మీరు ఆస్పిరిన్, రక్తం పలుచబెట్టే మందులు లేదా గుండె మందులు ఏవైనా తీసుకుంటున్నారా?"
                    },
                    0.9, "yesno",
                    category="medication: antiplatelets",
                    symptom=None
                ),
            ],
            weight=1.3
        )
        self.hybrid_symptom_nodes["Headache_Highblood pressure_legbleeding"] = headache_hbp_legbleeding

        # 5) Allergy_drymouth
        allergy_drymouth = HybridSymptomNode(
            "Allergy_drymouth",
            ["allergy", "dry mouth"],
            [
                make_question(
                    {
                        "en": "When did the dryness of the mouth start—before allergy symptoms or after starting medications?",
                        "hi": "मुँह में सूखापन कब से है— एलर्जी के लक्षणों से पहले शुरू हुआ या दवाएं शुरू करने के बाद?",
                        "gu": "મોઢામાં સૂકોપણું ક્યારેથી છે— શું તે એલર્જીના લક્ષણો પહેલાં શરૂ થયું કે દવા શરૂ કર્યા પછી?",
                        "te": "నోరు ఎండిపోవడం ఎప్పటి నుంచి ఉంది— అలర్జీ లక్షణాలకంటే ముందే మొదలైందా, లేక మందులు ప్రారంభించిన తర్వాతా?"
                    },
                    0.9, "yesno",
                    category="duration: allergy vs meds",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you have other dryness symptoms such as dry eyes, dry skin, or dry throat?",
                        "hi": "क्या आँखों, त्वचा या गले में भी सूखापन है?",
                        "gu": "શું તમને આંખોમાં, ત્વચામાં કે ગળામાં પણ સૂકોપણું છે?",
                        "te": "మీకు కళ్ల ఎండిపోవటం, చర్మం ఎండిపోవటం, లేదా గొంతు ఎండిపోవటం వంటి మరే ఇతర ఎండిపోయే లక్షణాలున్నాయా?"
                    },
                    0.9, "yesno",
                    category="dry skin",
                    symptom="dry skin"
                ),
                make_question(
                    {
                        "en": "Are you currently taking any medications for allergy, nasal congestion, blood pressure, or psychiatric conditions?",
                        "hi": "क्या आप वर्तमान में एलर्जी, नाक बंद होने, ब्लड प्रेशर या मानसिक रोग के लिए दवाएं ले रहे हैं?",
                        "gu": "શું તમે હાલમાં એલર્જી, નાકમાં જમાવ, બ્લડ પ્રેશર અથવા માનસિક બીમારી માટે કોઈ દવાઓ લઈ રહ્યા છો?",
                        "te": "మీరు ప్రస్తుతం అలర్జీ, ముక్కు దిబ్బడ, బ్లడ్ ప్రెజర్, లేదా మానసిక వ్యాధుల కోసం ఏమైనా మందులు తీసుకుంటున్నారా?"
                    },
                    0.9, "yesno",
                    category="medication: antihistamines/decongestants",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you feel excessively thirsty or have increased frequency of urination?",
                        "hi": "क्या आपको अत्यधिक प्यास लगती है या पेशाब की मात्रा/बार-बार पेशाब बढ़ गया है?",
                        "gu": "શું તમને બહુ તરસ લાગે છે અથવા વારંવાર/વધુ માત્રામાં મૂત્ર થતું હોય છે?",
                        "te": "మీకు ఎక్కువ దాహం వేయటం లేదా తరచుగా/ఎక్కువగా మూత్రం పోవడం జరుగుతున్నదా?"
                    },
                    0.9, "yesno",
                    category="frequent urination",
                    symptom="frequent urination",
                    risk_factor=True
                ),
                make_question(
                    {
                        "en": "Do you have nasal congestion, sneezing, or seasonal itching?",
                        "hi": "क्या नाक बंद होना, छींक आना या मौसम बदलने पर खुजली होती है?",
                        "gu": "શું તમને નાક બંધ થવું, છીંક આવવી અથવા ઋતુ બદલાય ત્યારે ખુજલી થાય છે?",
                        "te": "మీకు ముక్కు దిబ్బడ, తుమ్ములు, లేదా ऋతువులు మారినప్పుడు దురద వంటి సమస్యలున్నాయా?"
                    },
                    0.85, "yesno",
                    category="allergic_rhinitis_features",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Have you noticed difficulty swallowing, dental problems, cracked lips, or change in taste?",
                        "hi": "क्या निगलने में कठिनाई, दाँतों की समस्याएं, होंठ फटना या स्वाद में बदलाव महसूस हुआ है?",
                        "gu": "શું તમને ગળવાથી ગળવામાં તકલીફ, દાંતની સમસ્યાઓ, હોઠ ફાટવા અથવા સ્વાદમાં ફેરફાર જોવા મળ્યો છે?",
                        "te": "మీకు మింగటంలో ఇబ్బంది, దంత సమస్యలు, పెదాలు పగలటం, లేదా రుచిలో మార్పు గమనించారా?"
                    },
                    0.9, "yesno",
                    category="drymouth_complications",
                    symptom=None
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["Allergy_drymouth"] = allergy_drymouth

        # 6) Allergy_headache
        allergy_headache = HybridSymptomNode(
            "Allergy_headache",
            ["allergy", "headache"],
            [
                make_question(
                    {
                        "en": "When did the headache start, and is it related to allergy flares or exposure to dust, pollen, perfume, or weather change?",
                        "hi": "सिरदर्द कब शुरू हुआ, और क्या यह एलर्जी बढ़ने या धूल, पराग, परफ्यूम या मौसम बदलने के संपर्क से जुड़ा है?",
                        "gu": "માથાનો દુખાવો ક્યારે શરૂ થયો, અને શું તે એલર્જી વધી જવાથી અથવા ધૂળ, પરાગ, સુગંધ (પરફ્યુમ) અથવા હવામાન બદલાવના સંપર્ક સાથે સંબંધિત છે?",
                        "te": "తలనొప్పి ఎప్పుడు మొదలైంది, అది అలర్జీ ఎక్కువైనప్పుడు లేదా దుమ్ము, పుప్పొడి, సువాసన (పర్ఫ్యూమ్), వాతావరణ మార్పులకు ఎక్స్‌పోజర్ అయినప్పుడు వస్తుందా?"
                    },
                    0.9, "yesno",
                    category="duration: allergy",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Where exactly is the headache felt—forehead, around the eyes, cheeks, or generalized? Does it worsen on bending forward?",
                        "hi": "सिरदर्द ठीक कहाँ होता है— माथे पर, आँखों के आसपास, गालों में या पूरे सिर में? क्या झुकने पर दर्द बढ़ता है?",
                        "gu": "માથાનો દુખાવો ચોક્કસ ક્યાં થાય છે— કપાળ પર, આંખોની આજુબાજુ, ગાલોમાં કે સમગ્ર માથામાં? શું આગળ વંકાવાથી દુખાવો વધી જાય છે?",
                        "te": "తలనొప్పి కచ్చితంగా ఎక్కడ ఉంటుంది— నుదుటి భాగంలోనా, కళ్ల చుట్టూ, గడ్డాల దగ్గరా, లేక మొత్తం తలలోనా? ముందు వంగినప్పుడు నొప్పి పెరుగుతుందా?"
                    },
                    0.9, "yesno",
                    category="location: headache",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you have nasal symptoms like sneezing, nasal blockage, post-nasal drip, or watery discharge?",
                        "hi": "क्या छींक, नाक बंद होना, पोस्ट-नैसल ड्रिप या पानी जैसा स्राव होता है?",
                        "gu": "શું તમને છીંક આવવી, નાક બંધ થવું, ગળામાં પાછળથી પાણી જેવા સ્રાવ (પોસ્ટ-નેઝલ ડ્રિપ) કે પાણી જેવો વહેતો સ્રાવ થાય છે?",
                        "te": "మీకు తుమ్ములు, ముక్కు దిబ్బడ, గొంతులో వెనుక నుంచి ద్రవం జారిపోవటం (పోస్ట్ నాసల్ డ్రిప్) లేదా నీళ్ళలా కారే ముక్కు స్రావం ఉందా?"
                    },
                    0.9, "yesno",
                    category="sneezing",
                    symptom="sneezing"
                ),
                make_question(
                    {
                        "en": "Do you experience nausea, sensitivity to light, or throbbing pain suggesting migraine attacks?",
                        "hi": "क्या मितली, रोशनी से परेशानी, या धड़कन जैसा दर्द (माइग्रेन के संकेत) होता है?",
                        "gu": "શું તમને મિતલી, પ્રકાશથી તકલીફ અથવા ધબકારાઓ જેવી ધબકતી માથાના દુખાવાની લાગણી (માઇગ્રેન જેવા) થાય છે?",
                        "te": "మీకు వికారం, వెలుతురు తట్టుకోలేకపోవటం, లేదా గుండెల్లో కొట్టుకునేలా తడిమే తలనొప్పి (మైగ్రేన్ లాంటి) ఉంటుందా?"
                    },
                    0.9, "yesno",
                    category="nausea",
                    symptom="nausea"
                ),
                make_question(
                    {
                        "en": "Are you currently using any nasal sprays, antihistamines, or painkillers — and do they give relief?",
                        "hi": "क्या आप वर्तमान में नेज़ल स्प्रे, एंटीहिस्टामिन या दर्द निवारक ले रहे हैं — और क्या उनसे राहत मिलती है?",
                        "gu": "શું તમે હાલમાં નેઝલ સ્પ્રે, એન્ટિહિસ્ટામિન અથવા પેનકિલર લઈ રહ્યા છો— અને શું તેનાથી રાહત મળે છે?",
                        "te": "మీరు ప్రస్తుతం ముక్కు స్ప్రేలు, యాంటీహిస్టమిన్లు లేదా నొప్పి మందులు వాడుతున్నారా— వాటివల్ల ఉపశమనం కలుగుతున్నదా?"
                    },
                    0.85, "yesno",
                    category="medication: allergy",
                    symptom=None
                ),
                make_question(
                    {
                        "en": "Do you have fever, facial tenderness, ear pressure, or a recent upper respiratory infection?",
                        "hi": "क्या बुखार, चेहरे पर दबाने से दर्द, कान में दबाव, या हाल ही में ऊपरी श्वसन संक्रमण हुआ है?",
                        "gu": "શું તમને તાવ, ચહેરા પર દબાવવાથી દુખાવો, કાનમાં દબાણ કે તાજેતરમાં ઉપરના શ્વાસ માર્ગનો ચેપ થયો છે?",
                        "te": "మీకు జ్వరం, ముఖాన్ని నొక్కితే నొప్పి, చెవిలో ఒత్తిడి, లేదా ఇటీవలి పై శ్వాసనాళ ఇన్ఫెక్షన్ ఉందా?"
                    },
                    0.9, "yesno",
                    category="fever",
                    symptom="fever",
                    risk_factor=True
                ),
            ],
            weight=1.2
        )
        self.hybrid_symptom_nodes["Allergy_headache"] = allergy_headache










    def build_gi_question_sets(self):
        """Define structured GI follow-up question sets for the stomach pain hybrid."""
        question_sets = {
            "G1": [
                {
                    "symptom": "pain_location",
                    "question": {
                        "en": "Do you have pain in the upper or lower abdomen?",
                        "hi": "क्या दर्द पेट के ऊपरी हिस्से में है या निचले हिस्से में?",
                    },
                    "options": ["upper", "lower", "both", "unsure"],
                    "weight": 0.9,
                    "type": "choice",
                },
                {
                    "symptom": "meal_worse",
                    "question": {
                        "en": "Does it worsen after meals?",
                        "hi": "क्या दर्द खाने के बाद बढ़ता है?",
                    },
                    "options": ["yes", "no", "sometimes", "only after spicy–oily food"],
                    "weight": 0.9,
                    "type": "choice",
                },
                {
                    "symptom": "pain_nature",
                    "question": {
                        "en": "Nature of pain? cramping or stabbing?",
                        "hi": "दर्द कैसा है—ऐंठन जैसा या चुभन जैसा?",
                    },
                    "options": ["cramping", "stabbing", "burning", "dull ache"],
                    "weight": 0.8,
                    "type": "choice",
                },
                {
                    "symptom": "blood_stools",
                    "question": {
                        "en": "Any blood in vomit or black, tarry stools?",
                        "hi": "क्या खून की उल्टी हुई है या काला, चिपचिपा मल दिखा?",
                    },
                    "options": ["yes", "no", "not sure"],
                    "weight": 0.9,
                    "type": "choice",
                },
                {
                    "symptom": "frequent_foods",
                    "question": {
                        "en": "Do you take painkillers, coffee/tea, or very spicy food often?",
                        "hi": "क्या आप अक्सर दर्द निवारक, कॉफी/चाय या बहुत मसालेदार खाना लेते हैं?",
                    },
                    "options": ["daily", "few times a week", "rarely", "never"],
                    "weight": 0.8,
                    "type": "choice",
                },
            ],
            "G2": [
                {
                    "symptom": "burning_chest",
                    "question": {
                        "en": "Is it burning in the chest or sour reflux to the throat?",
                        "hi": "सीने में जलन या खट्टी डकार/मुँह तक खटास आती है?",
                    },
                    "options": ["yes often", "occasional", "no"],
                    "weight": 0.9,
                    "type": "choice",
                },
                {
                    "symptom": "pain_relation",
                    "question": {
                        "en": "Is the pain related to empty stomach vs after spicy/tea/coffee?",
                        "hi": "खाली पेट या मसालेदार/चाय-कॉफी के बाद दर्द बढ़ता है?",
                    },
                    "options": ["empty stomach", "after spicy/tea/coffee", "both", "none"],
                    "weight": 0.8,
                    "type": "choice",
                },
                {
                    "symptom": "night_antacid",
                    "question": {
                        "en": "Night-time symptoms or relief with antacid/gel?",
                        "hi": "रात में ज़्यादा होता है या एंटासिड से आराम मिलता है?",
                    },
                    "options": ["night-time yes", "antacid helps", "no relief"],
                    "weight": 0.8,
                    "type": "choice",
                },
                {
                    "symptom": "longterm_nsaids",
                    "question": {
                        "en": "Any long-term painkiller use (diclofenac, ibuprofen)?",
                        "hi": "क्या लंबे समय से दर्द निवारक (जैसे डाइक्लोफेनैक, इबुप्रोफेन) लेते हैं?",
                    },
                    "options": ["daily", "weekly", "occasionally", "never"],
                    "weight": 0.8,
                    "type": "choice",
                },
                {
                    "symptom": "stress_meals",
                    "question": {
                        "en": "Recent stress, irregular meals, outside food?",
                        "hi": "हाल में तनाव, अनियमित भोजन, बाहर का तला-भुना ज्यादा?",
                    },
                    "options": ["high", "moderate", "low"],
                    "weight": 0.7,
                    "type": "choice",
                },
            ],
            "G3": [
                {
                    "symptom": "stool_change",
                    "question": {
                        "en": "Bloating with change in stool—loose/constipation?",
                        "hi": "पेट फूलना और मल में बदलाव—ढीला/कब्ज?",
                    },
                    "options": ["loose", "constipation", "alternating", "normal"],
                    "weight": 0.9,
                    "type": "choice",
                },
                {
                    "symptom": "pain_relief",
                    "question": {
                        "en": "Pain better after passing gas or stool?",
                        "hi": "गैस या शौच के बाद दर्द कम होता है?",
                    },
                    "options": ["yes", "no", "partial"],
                    "weight": 0.8,
                    "type": "choice",
                },
                {
                    "symptom": "fever_urinary",
                    "question": {
                        "en": "Any fever, burning urination, or urinary frequency?",
                        "hi": "बुखार, पेशाब में जलन या बार-बार पेशाब?",
                    },
                    "options": ["yes", "no"],
                    "weight": 0.8,
                    "type": "choice",
                },
                {
                    "symptom": "menstrual_relation",
                    "question": {
                        "en": "For menstruating patients: relation to periods?",
                        "hi": "माहवारी से पहले/दौरान दर्द बढ़ता है?",
                    },
                    "options": ["before", "during", "after", "not applicable"],
                    "weight": 0.7,
                    "type": "choice",
                },
                {
                    "symptom": "recent_travel_food",
                    "question": {
                        "en": "Recent outside food, unsafe water, travel?",
                        "hi": "हाल में बाहर का खाना, असुरक्षित पानी, यात्रा?",
                    },
                    "options": ["yes", "no"],
                    "weight": 0.8,
                    "type": "choice",
                },
            ],
            "G4": [
                {
                    "symptom": "episodes",
                    "question": {
                        "en": "How many episodes and when last?",
                        "hi": "कितनी बार हुआ और आख़िरी बार कब?",
                    },
                    "options": None,
                    "weight": 0.9,
                    "type": "text",
                },
                {
                    "symptom": "dizziness",
                    "question": {
                        "en": "Dizziness, fainting, paleness?",
                        "hi": "चक्कर, बेहोशी, पीलापन?",
                    },
                    "options": ["yes", "no"],
                    "weight": 0.9,
                    "type": "choice",
                },
                {
                    "symptom": "weight_loss",
                    "question": {
                        "en": "Unintentional weight loss or appetite drop?",
                        "hi": "बिना कोशिश वजन घटा या भूख कम हुई?",
                    },
                    "options": ["yes", "no"],
                    "weight": 0.8,
                    "type": "choice",
                },
                {
                    "symptom": "liver_disease",
                    "question": {
                        "en": "Any liver disease, jaundice, alcohol use?",
                        "hi": "लीवर की बीमारी, पीलिया, शराब का सेवन?",
                    },
                    "options": ["yes", "no"],
                    "weight": 0.8,
                    "type": "choice",
                },
                {
                    "symptom": "current_meds",
                    "question": {
                        "en": "Current meds: blood thinners/NSAIDs/steroids?",
                        "hi": "कौन-सी दवाएँ—खून पतला करने वाली/दर्द निवारक/स्टेरॉइड?",
                    },
                    "options": None,
                    "weight": 0.8,
                    "type": "text",
                },
            ],
            "G5": [
                {
                    "symptom": "tea_coffee_freq",
                    "question": {
                        "en": "Frequency of tea/coffee & timing vs meals?",
                        "hi": "चाय-कॉफी कितनी बार और खाने के आसपास लेते हैं?",
                    },
                    "options": ["≥3/day", "1–2/day", "rarely"],
                    "weight": 0.7,
                    "type": "choice",
                },
                {
                    "symptom": "smoking_alcohol",
                    "question": {
                        "en": "Smoking/chewing tobacco or alcohol?",
                        "hi": "धूम्रपान/तंबाकू या शराब?",
                    },
                    "options": ["yes daily", "occasional", "never"],
                    "weight": 0.8,
                    "type": "choice",
                },
                {
                    "symptom": "family_history",
                    "question": {
                        "en": "Family history of ulcer/acid issues?",
                        "hi": "परिवार में अल्सर/एसिडिटी का इतिहास?",
                    },
                    "options": ["yes", "no", "unknown"],
                    "weight": 0.7,
                    "type": "choice",
                },
                {
                    "symptom": "hpylori_test",
                    "question": {
                        "en": "Prior H. pylori test or treatment?",
                        "hi": "पहले एच-पाइलोरी की जाँच/इलाज हुआ?",
                    },
                    "options": ["positive treated", "tested negative", "never tested"],
                    "weight": 0.8,
                    "type": "choice",
                },
                {
                    "symptom": "pain_score",
                    "question": {
                        "en": "Pain score 0–10 and impact on work/sleep?",
                        "hi": "दर्द 0–10 और काम/नींद पर असर?",
                    },
                    "options": None,
                    "weight": 0.8,
                    "type": "text",
                },
            ],
        }
        self.gi_question_lookup = {}
        for set_name, questions in question_sets.items():
            for question in questions:
                symptom_key = question.get("symptom")
                if symptom_key:
                    self.gi_question_lookup[(set_name, symptom_key)] = question
        return question_sets

    def reset_gi_followup_state(self):
        """Reset stored responses and asked sets for the GI hybrid flow."""
        self.gi_followup_responses = {}
        self.asked_gi_sets = set()

    def normalize_gi_answer(self, symptom_key: str, answer: str) -> str:
        """Normalize free-form answers into canonical buckets for trigger logic."""
        if answer is None:
            return ""
        ans_text = str(answer).strip()
        ans = ans_text.lower()

        if symptom_key == "pain_location":
            if "upper" in ans:
                return "upper"
            if "lower" in ans:
                return "lower"
            if "both" in ans:
                return "both"
            return "unsure"

        if symptom_key == "meal_worse":
            if "only" in ans and ("spicy" in ans or "oily" in ans):
                return "only after spicy–oily food"
            if "yes" in ans:
                return "yes"
            if "some" in ans:
                return "sometimes"
            if "spicy" in ans or "oily" in ans:
                return "only after spicy–oily food"
            return "no"

        if symptom_key == "pain_nature":
            if "cramp" in ans:
                return "cramping"
            if "stab" in ans:
                return "stabbing"
            if "burn" in ans:
                return "burning"
            if "dull" in ans:
                return "dull ache"
            return ans or "dull ache"

        if symptom_key == "frequent_foods":
            if "daily" in ans or "every day" in ans or ans == "yes":
                return "daily"
            if "few" in ans or "sometimes" in ans or "weekly" in ans:
                return "few times a week"
            if "rare" in ans:
                return "rarely"
            return "never"

        return ans_text

    def ask_gi_question_set(self, set_name: str, detail_target: Dict[str, str], max_questions: int = 3):
        """Ask a defined GI question set and store responses for later triggers."""
        questions = self.gi_question_sets.get(set_name, [])
        if not questions or set_name in self.asked_gi_sets:
            return []

        print(f"\n{'=' * 60}")
        print(f"GI Follow-up Questions: {set_name}")
        print(f"{'=' * 60}")

        responses = []
        for question in questions[:max_questions]:
            q_text = question["question"]
            display_text = self.get_display_text(q_text)
            detail_key = f"[{set_name}] {display_text}"

            if question["type"] == "choice":
                options = ", ".join(question["options"] or [])
                raw_answer = input(f"{display_text}\nOptions: {options}\nAnswer: ").strip()
                normalized = self.normalize_gi_answer(question["symptom"], raw_answer)
                self.gi_followup_responses[question["symptom"]] = normalized
                detail_target[detail_key] = normalized
                responses.append((q_text, normalized, question["weight"]))
            elif question["type"] == "text":
                raw_answer = input(f"{display_text}\nAnswer: ").strip()
                self.gi_followup_responses[question["symptom"]] = raw_answer
                detail_target[detail_key] = raw_answer
                responses.append((q_text, raw_answer, question["weight"]))
            else:  # yes/no handling treated as choice but capture boolean for confidence
                raw_answer = input(f"{display_text} (yes/no): ").strip().lower()
                is_positive = raw_answer.startswith("y")
                normalized = "yes" if is_positive else "no"
                self.gi_followup_responses[question["symptom"]] = normalized
                detail_target[detail_key] = normalized
                responses.append((q_text, is_positive, question["weight"]))

        self.asked_gi_sets.add(set_name)
        return responses

    def determine_gi_triggers(self, responses: Dict[str, str]) -> List[str]:
        """Use gathered responses to decide which GI question sets to ask next."""
        triggers = []

        def add_trigger(name: str):
            if name not in triggers:
                triggers.append(name)

        pain_location = responses.get("pain_location")
        meal_worse = responses.get("meal_worse")
        pain_nature = responses.get("pain_nature")

        if responses.get("blood_stools") == "yes":
            add_trigger("G4")

        if pain_location == "upper":
            add_trigger("G2")
        elif pain_location == "lower":
            add_trigger("G3")
        elif pain_location == "both":
            add_trigger("G2")
            add_trigger("G3")
        else:
            if meal_worse in {"yes", "sometimes", "only after spicy–oily food"}:
                add_trigger("G2")

        if pain_nature == "cramping":
            add_trigger("G3")

        frequent_foods = responses.get("frequent_foods")
        if frequent_foods in {"daily", "few times a week"}:
            add_trigger("G5")

        if responses.get("longterm_nsaids") in {"daily", "weekly"} or responses.get("night_antacid") == "night-time yes":
            if responses.get("blood_stools") == "yes":
                add_trigger("G4")
            else:
                add_trigger("G5")

        if responses.get("fever_urinary") == "yes":
            add_trigger("G4")

        menstrual_relation = responses.get("menstrual_relation")
        if menstrual_relation and menstrual_relation != "not applicable":
            add_trigger("G5")

        return triggers

    def run_gi_dynamic_followup(self, detail_target: Dict[str, str]):
        """Run the multi-stage GI follow-up flow starting with baseline G1."""
        pending_sets = []

        if "G1" not in self.asked_gi_sets:
            self.ask_gi_question_set("G1", detail_target)
            pending_sets.extend(self.determine_gi_triggers(self.gi_followup_responses))

        while pending_sets:
            next_set = pending_sets.pop(0)
            if next_set in self.asked_gi_sets:
                continue

            self.ask_gi_question_set(next_set, detail_target)
            for triggered in self.determine_gi_triggers(self.gi_followup_responses):
                if triggered not in pending_sets and triggered not in self.asked_gi_sets:
                    pending_sets.append(triggered)

        print("\nGI dynamic follow-up completed.")

    # ───────── Session-driven GI helper utilities ─────────
    def extract_gi_followup_state(self, conversation_history: List[Dict[str, str]]):
        """Derive asked sets, asked questions, and normalized responses from history."""
        responses: Dict[str, str] = {}
        asked_sets: Set[str] = set()
        asked_questions: Dict[str, Set[str]] = defaultdict(set)
        prefix = f"{self.gi_hybrid_key}_"

        for entry in conversation_history:
            category = entry.get("category")
            if not isinstance(category, str) or not category.startswith(prefix):
                continue

            remainder = category[len(prefix):]
            if not remainder:
                continue

            parts = remainder.split("_", 1)
            set_name = parts[0]
            asked_sets.add(set_name)
            symptom_key = parts[1] if len(parts) > 1 else None

            if not symptom_key:
                continue

            asked_questions[set_name].add(symptom_key)
            response_text = entry.get("response")
            if response_text is None:
                continue

            question_meta = self.gi_question_lookup.get((set_name, symptom_key))
            if not question_meta:
                continue

            if question_meta.get("type") == "choice":
                responses[symptom_key] = self.normalize_gi_answer(symptom_key, response_text)
            else:
                responses[symptom_key] = str(response_text).strip()

        return responses, asked_sets, asked_questions

    def _augment_gi_counts_from_categories(
        self,
        asked_sets: Set[str],
        asked_questions: Dict[str, Set[str]],
        asked_categories: Set[str],
    ) -> None:
        """Include any previously asked GI questions tracked only by category IDs.

        Some sessions may persist asked question categories even before the
        conversation history has the corresponding response entry. When that
        happens we still need to honour the per-set quota, otherwise the engine
        might think fewer questions were asked and keep re-scheduling them.
        """

        if not asked_categories:
            return

        prefix = f"{self.gi_hybrid_key}_"
        for category in asked_categories:
            if not category.startswith(prefix):
                continue

            remainder = category[len(prefix):]
            if not remainder:
                continue

            set_name, _, symptom_key = remainder.partition("_")
            if not set_name:
                continue

            asked_sets.add(set_name)
            if symptom_key:
                asked_questions.setdefault(set_name, set()).add(symptom_key)

    def next_gi_set_to_ask(self,
                           responses: Dict[str, str],
                           asked_sets: Set[str],
                           asked_questions: Dict[str, Set[str]],
                           limit_per_set: int = 3) -> Optional[str]:
        """Determine the next GI question set that still needs coverage."""
        g1_questions = self.gi_question_sets.get("G1", [])
        g1_target = min(limit_per_set, len(g1_questions))
        if len(asked_questions.get("G1", set())) < g1_target:
            return "G1"

        trigger_order = ["G4", "G2", "G3", "G5"]
        triggered_sets = self.determine_gi_triggers(responses)

        for set_name in trigger_order:
            if set_name not in triggered_sets:
                continue
            questions = self.gi_question_sets.get(set_name, [])
            target = min(limit_per_set, len(questions))
            if len(asked_questions.get(set_name, set())) < target:
                return set_name

        for set_name in triggered_sets:
            questions = self.gi_question_sets.get(set_name, [])
            target = min(limit_per_set, len(questions))
            if len(asked_questions.get(set_name, set())) < target:
                return set_name

        return None

    def render_gi_followup_questions(self,
                                     set_name: str,
                                     asked_questions: Dict[str, Set[str]],
                                     asked_categories: Set[str],
                                     limit_per_set: int = 3) -> List[Dict[str, str]]:
        """Prepare question payloads for the given GI set respecting asked limits."""
        questions = []
        available = self.gi_question_sets.get(set_name, [])
        already_asked = asked_questions.get(set_name, set())
        target_count = min(limit_per_set, len(available))
        remaining_quota = max(0, target_count - len(already_asked))

        if remaining_quota == 0:
            return questions

        for question in available:
            if len(questions) >= remaining_quota:
                break

            symptom_key = question.get("symptom")
            if not symptom_key or symptom_key in already_asked:
                continue

            category = f"{self.gi_hybrid_key}_{set_name}_{symptom_key}"
            if categories_conflict(category, asked_categories):
                continue

            payload = {
                "en": question["question"]["en"],
                "hi": question["question"]["hi"],
                "category": category,
                "symptom": symptom_key,
                "risk_factor": question.get("risk_factor", False),
            }

            if question.get("type") == "choice" and question.get("options"):
                payload["options"] = question["options"]

            questions.append(payload)

        return questions

    def prepare_gi_followup_questions(self,
                                      conversation_history: List[Dict[str, str]],
                                      asked_categories: Set[str],
                                      limit_per_set: int = 3) -> List[Dict[str, str]]:
        """Generate the next batch of GI follow-up questions for the hybrid."""
        responses, asked_sets, asked_questions = self.extract_gi_followup_state(conversation_history)
        self._augment_gi_counts_from_categories(asked_sets, asked_questions, asked_categories)
        next_set = self.next_gi_set_to_ask(responses, asked_sets, asked_questions, limit_per_set)
        if not next_set:
            return []

        questions = self.render_gi_followup_questions(next_set, asked_questions, asked_categories, limit_per_set)
        if questions:
            asked_sets.add(next_set)
        return questions

    def build_disease_knowledge(self):
        """Build enhanced disease knowledge base with original diseases and weighted symptoms"""

        self.diseases = [
            # Pulmonary Tuberculosis
            DiseaseNode("Pulmonary Tuberculosis", 
                ["fever", "cough > 2 wks", "night sweats", "weight loss", "fatigue", 
                    "anorexia", "hemoptysis", "chest pain", "chestpain_breathlessness",
                    "fever_cough", "fever_headache", "fever_fatigue"],
                [
                    ({"en": "Have you had close contact with anyone diagnosed with tuberculosis (TB)?",
                    "hi": "क्या आपका किसी क्षय रोग से पीड़ित व्यक्ति से निकट संपर्क रहा है?"}, 0.9, "yesno"),
                    ({"en": "Do you have HIV or any condition that weakens immunity?",
                    "hi": "क्या आपको एचआईवी है या कोई ऐसी समस्या है जिससे रोग-प्रतिरोधक क्षमता कमज़ोर हो जाती है?"}, 0.9, "yesno"),
                    ({"en": "Have these symptoms lasted for more than two weeks?",
                    "hi": "क्या ये लक्षण दो सप्ताह से अधिक समय से चल रहे हैं?"}, 0.8, "yesno")
                ],
                {
                    "fever": 0.8, "cough > 2 wks": 1.0, "night sweats": 0.9,
                    "weight loss": 0.8, "hemoptysis": 1.0, "fatigue": 0.5,
                    "anorexia": 0.6, "chest pain": 0.7,
                    "chestpain_breathlessness": 0.8, "fever_cough": 0.8,
                    "fever_headache": 0.7, "fever_fatigue": 0.7
                }),

            # Pneumonia
            DiseaseNode("Pneumonia",
                ["fever", "cough", "chest pain", "dyspnea", "productive cough", 
                    "fatigue", "chills", "chestpain_breathlessness", "fever_cough", 
                    "cough_dyspnea", "cough_chest_pain"],
                [
                    ({"en": "Have you noticed rusty or blood-tinged sputum (phlegm)?",
                    "hi": "क्या आपने जंग जैसा या खून मिला हुआ बलगम देखा है?"}, 0.9, "yesno"),
                    ({"en": "Do you have sharp chest pain that worsens when you breathe (pleuritic pain)?",
                    "hi": "क्या आपको सांस लेने पर बढ़ने वाला तेज़ सीने का दर्द होता है?"}, 0.8, "yesno"),
                    ({"en": "Have you had a recent chest or throat infection?",
                    "hi": "क्या आपको हाल ही में छाती या गले का कोई संक्रमण हुआ है?"}, 0.6, "yesno")
                ],
                {
                    "fever": 0.9, "cough": 0.8, "chest pain": 0.7,
                    "dyspnea": 0.8, "productive cough": 0.9, "chills": 0.7,
                    "fatigue": 0.5,
                    "chestpain_breathlessness": 0.8, "fever_cough": 0.8,
                    "cough_dyspnea": 0.8, "cough_chest_pain": 0.7
                }),

            # Dengue Fever
            DiseaseNode("Dengue Fever",
                ["fever", "headache", "retro-orbital pain", "myalgia", "rash", 
                    "nausea/vomiting", "bleeding manifestations", "fever_headache", 
                    "headache_vomiting"],
                [
                    ({"en": "Have you noticed small red spots, gum bleeding, or easy bruising on the skin?",
                    "hi": "क्या आपको छोटे लाल धब्बे, मसूड़ों से खून आना, या त्वचा पर आसानी से नीला पड़ना दिखाई दिया है?"}, 0.9, "yesno"),
                    ({"en": "Have you recently been to an area where dengue is common?",
                    "hi": "क्या आप हाल ही में ऐसे क्षेत्र में गए हैं जहाँ डेंगू आम है?"}, 0.8, "yesno"),
                    ({"en": "Are you having severe abdominal pain?",
                    "hi": "क्या आपको तेज़ पेट दर्द हो रहा है?"}, 0.7, "yesno")
                ],
                {
                    "fever": 1.0, "headache": 0.8, "retro-orbital pain": 0.9,
                    "myalgia": 0.7, "rash": 0.7, "nausea/vomiting": 0.6,
                    "bleeding manifestations": 0.9,
                    "fever_headache": 0.8, "headache_vomiting": 0.7
                }),

            # Malaria
            DiseaseNode("Malaria",
                ["fever", "chills", "headache", "myalgia", "jaundice", 
                    "splenomegaly", "nausea/vomiting", "fever_chills_cough", 
                    "fever_headache", "fever_vomiting"],
                [
                    ({"en": "Do you get fever in cycles (every 48–72 hours)?",
                    "hi": "क्या आपको चक्रीय बुखार आता है—हर 48–72 घंटे में?"}, 1.0, "yesno"),
                    ({"en": "Have you recently traveled to or lived in a malaria-affected area?",
                    "hi": "क्या आप हाल ही में मलेरिया प्रभावित क्षेत्र में गए हैं या वहाँ रहे हैं?"}, 0.9, "yesno"),
                    ({"en": "Have you noticed very dark-colored urine?",
                    "hi": "क्या आपने बहुत गहरे रंग का पेशाब देखा है?"}, 0.8, "yesno")
                ],
                {
                    "fever": 1.0, "chills": 0.9, "headache": 0.7,
                    "myalgia": 0.6, "jaundice": 0.8, "splenomegaly": 0.7,
                    "nausea/vomiting": 0.5,
                    "fever_chills_cough": 0.8, "fever_headache": 0.7, "fever_vomiting": 0.7
                }),

            # Typhoid Fever
            DiseaseNode("Typhoid Fever",
                ["fever", "headache", "abdominal pain", "constipation", "rose spots", 
                    "malaise", "dry cough", "fever_headache", "abdominal_pain_diarrhea"],
                [
                    ({"en": "Does the fever rise a little more each day (step-ladder pattern)?",
                    "hi": "क्या बुखार रोज़ थोड़ा-थोड़ा बढ़ता है, यानी सीढ़ीनुमा पैटर्न?"}, 0.9, "yesno"),
                    ({"en": "Have you recently eaten food or drunk water that might have been contaminated?",
                    "hi": "क्या आपने हाल में ऐसा भोजन या पानी लिया है जो दूषित हो सकता था?"}, 0.8, "yesno"),
                    ({"en": "Has any doctor told you that your pulse was slower than expected during fever (relative bradycardia)?",
                    "hi": "क्या किसी डॉक्टर ने बताया कि बुखार के दौरान आपकी नाड़ी सामान्य से धीमी थी, जिसे रिलेटिव ब्रैडिकार्डिया कहा जाता है?"}, 0.7, "yesno")
                ],
                {
                    "fever": 1.0, "headache": 0.8, "abdominal pain": 0.8,
                    "constipation": 0.7, "rose spots": 0.9, "malaise": 0.5,
                    "dry cough": 0.6,
                    "fever_headache": 0.7, "abdominal_pain_diarrhea": 0.8
                }),

            # Hepatitis A/E
            DiseaseNode("Hepatitis A/E",
                ["jaundice", "fever", "anorexia", "dark urine", "pale stools", 
                    "abdominal pain", "fatigue", "abdominal_pain_vomiting", "fever_fatigue"],
                [
                    ({"en": "Have you recently consumed food or water that might have been contaminated?",
                    "hi": "क्या आपने हाल में ऐसा भोजन या पानी लिया है जो दूषित हो सकता था?"}, 0.8, "yesno"),
                    ({"en": "Do you have joint pains or muscle aches?",
                    "hi": "क्या आपको जोड़ों में दर्द या मांसपेशियों में दर्द है?"}, 0.6, "yesno"),
                    ({"en": "Have you recently traveled to an area where hepatitis is common?",
                    "hi": "क्या आप हाल में ऐसे क्षेत्र में गए हैं जहाँ हेपेटाइटिस आम है?"}, 0.7, "yesno")
                ],
                {
                    "jaundice": 1.0, "fever": 0.7, "anorexia": 0.8,
                    "dark urine": 0.9, "pale stools": 0.9, "abdominal pain": 0.7,
                    "fatigue": 0.6,
                    "abdominal_pain_vomiting": 0.8, "fever_fatigue": 0.7
                }),

            # Influenza (Flu)
            DiseaseNode("Influenza (Flu)",
                ["fever", "headache", "myalgia", "sore throat", "dry cough", "fatigue", 
                    "nasal congestion", "fever_headache", "fever_cough", "fever_fatigue_cough"],
                [
                    ({"en": "Did your symptoms begin suddenly?", "hi": "क्या आपके लक्षण अचानक शुरू हुए थे?"}, 0.8, "yesno"),
                    ({"en": "Do your symptoms usually occur in the winter season?", "hi": "क्या आपके लक्षण अधिकतर सर्दियों में होते हैं?"}, 0.7, "yesno"),
                    ({"en": "Have you received the flu vaccine this season?", "hi": "क्या आपने इस सीज़न में फ्लू का टीका लगवाया है?"}, 0.5, "yesno")
                ],
                {
                    "fever": 0.9, "headache": 0.7, "myalgia": 0.8,
                    "sore throat": 0.6, "dry cough": 0.7, "fatigue": 0.8,
                    "nasal congestion": 0.6,
                    "fever_headache": 0.7, "fever_cough": 0.8, "fever_fatigue_cough": 0.8
                }),

            # COVID-19
            DiseaseNode("COVID-19",
                ["fever", "dry cough", "dyspnea", "myalgia", "loss of taste", 
                    "loss of smell", "fatigue", "fever_cough", "cough_dyspnea_chestpain"],
                [
                    ({"en": "Have you been in close contact with a person who had COVID-19?",
                    "hi": "क्या आपका हाल ही में किसी कोविड-19 रोगी से निकट संपर्क हुआ है?"}, 0.8, "yesno"),
                    ({"en": "Have you traveled recently to an area with a COVID-19 outbreak?",
                    "hi": "क्या आप हाल ही में ऐसे क्षेत्र में गए हैं जहाँ कोविड-19 का प्रकोप था?"}, 0.6, "yesno"),
                    ({"en": "Has a chest CT scan shown findings suggestive of COVID-19?",
                    "hi": "क्या आपकी छाती की सीटी स्कैन में कोविड-19 से संबंधित संकेत मिले हैं?"}, 0.7, "yesno")
                ],
                {
                    "fever": 0.8, "dry cough": 0.8, "dyspnea": 0.8,
                    "myalgia": 0.6, "loss of taste": 0.9, "loss of smell": 0.9,
                    "fatigue": 0.7,
                    "fever_cough": 0.8, "cough_dyspnea_chestpain": 0.8
                }),

            # Bronchitis
            DiseaseNode("Bronchitis",
                    ["cough < 2 wks", "productive cough", "chest pain", "wheezing", 
                        "sore throat", "low-grade fever", "fever_cough", 
                        "chestpain_breathlessness", "cough_chest_pain"],
                    [
                        ({"en": "Do you have a history of smoking?", "hi": "क्या आपका धूम्रपान का इतिहास है?"}, 0.7, "yesno"),
                        ({"en": "Do you have post-nasal drip (mucus dripping down the throat)?", "hi": "क्या आपको पोस्ट-नेज़ल ड्रिप होता है, यानी नाक से गले की ओर बलगम बहता है?"}, 0.6, "yesno"),
                        ({"en": "Do you have seasonal allergies?", "hi": "क्या आपको मौसमी एलर्जी होती है?"}, 0.5, "yesno")
                    ],
                    {
                        "cough < 2 wks": 0.9, "productive cough": 0.8, "chest pain": 0.6,
                        "wheezing": 0.7, "sore throat": 0.5, "low-grade fever": 0.6,
                        "fever_cough": 0.8, "chestpain_breathlessness": 0.7, "cough_chest_pain": 0.7
                    }),

            # Cholera
            DiseaseNode("Cholera",
                    ["diarrhea", "vomiting", "abdominal pain", "dehydration", 
                        "rice-water stools", "thirst", "oliguria", "diarrhea_vomiting", 
                        "abdominal_pain_diarrhea", "abdominalpain_diarrhea_vomiting"],
                    [
                        ({"en": "Have you recently eaten raw or undercooked seafood?", "hi": "क्या आपने हाल में कच्चा या कम पका समुद्री भोजन खाया है?"}, 0.7, "yesno"),
                        ({"en": "Are there signs of severe dehydration (very dry mouth, little urine, dizziness)?", "hi": "क्या गंभीर निर्जलीकरण के लक्षण हैं, जैसे बहुत सूखा मुँह, बहुत कम पेशाब, चक्कर?"}, 0.9, "yesno"),
                        ({"en": "Do you live in or have you visited a cholera-affected area?", "hi": "क्या आप हैजा प्रभावित क्षेत्र में रहते हैं या वहाँ गए हैं?"}, 0.8, "yesno")
                    ],
                    {
                        "diarrhea": 1.0, "vomiting": 0.8, "abdominal pain": 0.6,
                        "dehydration": 0.9, "rice-water stools": 1.0, "thirst": 0.7,
                        "oliguria": 0.8,
                        "diarrhea_vomiting": 0.9, "abdominal_pain_diarrhea": 0.8, "abdominalpain_diarrhea_vomiting": 0.9
                    }),

            # Genital Herpes
            DiseaseNode("Genital Herpes",
                    ["fever", "painful genital ulcers", "dysuria", "inguinal lymphadenopathy", 
                        "malaise", "myalgia"],
                    [
                        ({"en": "Do you have multiple painful blisters or ulcers in the genital area?",
                        "hi": "क्या जननांग क्षेत्र में कई दर्दनाक छाले या घाव हैं?"}, 0.9, "yesno"),
                        ({"en": "Have you had similar episodes in the past?",
                        "hi": "क्या आपको पहले भी ऐसे एपिसोड हुए हैं?"}, 0.8, "yesno"),
                        ({"en": "Have you had sexual contact with a partner who might be infected?",
                        "hi": "क्या आपका ऐसे साथी के साथ यौन संपर्क हुआ है जो संक्रमित हो सकता है?"}, 0.8, "yesno")
                    ],
                    {
                        "fever": 0.7, "painful genital ulcers": 1.0, "dysuria": 0.8,
                        "inguinal lymphadenopathy": 0.7, "malaise": 0.6, "myalgia": 0.5
                    }),

            # Chikungunya
            DiseaseNode("Chikungunya",
                    ["fever", "polyarthralgia", "headache", "myalgia", "rash", "conjunctivitis", "fatigue"],
                    [
                        ({"en": "Are you having disabling joint pains?", "hi": "क्या आपको चलने-फिरने में बाधा देने वाला जोड़ों का तेज़ दर्द है?"}, 0.9, "yesno"),
                        ({"en": "Have you traveled recently to an area with a chikungunya outbreak?", "hi": "क्या आप हाल में ऐसे क्षेत्र में गए हैं जहाँ चिकनगुनिया का प्रकोप है?"}, 0.8, "yesno"),
                        ({"en": "Have you been bitten by mosquitoes during the day?", "hi": "क्या आपको दिन में मच्छरों ने काटा है?"}, 0.7, "yesno")
                    ],
                    {
                        "fever": 0.8, "polyarthralgia": 1.0, "headache": 0.7,
                        "myalgia": 0.7, "rash": 0.7, "conjunctivitis": 0.6,
                        "fatigue": 0.6
                    }),

            # Dysentery
            DiseaseNode("Dysentery",
                    ["diarrhea", "blood in stools", "fever", "abdominal pain", "tenesmus", 
                        "mucus in stools", "abdominal_pain_diarrhea", "diarrhea_headache", 
                        "abdominalpain_diarrhea_vomiting"],
                    [
                        ({"en": "Have you recently taken antibiotics?", "hi": "क्या आपने हाल में एंटीबायोटिक दवाएँ ली हैं?"}, 0.6, "yesno"),
                        ({"en": "Do you have a history of food poisoning?", "hi": "क्या आपको भोजन विषाक्तता का इतिहास रहा है?"}, 0.7, "yesno"),
                        ({"en": "Do you have any signs of dehydration (dry mouth, low urine, dizziness)?", "hi": "क्या आपको निर्जलीकरण के लक्षण हैं, जैसे मुँह सूखना, कम पेशाब, चक्कर?"}, 0.8, "yesno")
                    ],
                    {
                        "diarrhea": 1.0, "blood in stools": 1.0, "fever": 0.7,
                        "abdominal pain": 0.8, "tenesmus": 0.7, "mucus in stools": 0.7,
                        "abdominal_pain_diarrhea": 0.8, "diarrhea_headache": 0.7, "abdominalpain_diarrhea_vomiting": 0.8
                    }),

            # Urinary Tract Infection
            DiseaseNode("Urinary Tract Infection",
                    ["dysuria", "urgency", "frequency", "suprapubic pain", "cloudy urine", "hematuria"],
                    [
                        ({"en": "Do you also have fever or chills?", "hi": "क्या आपको बुखार या ठंड भी लगती है?"}, 0.7, "yesno"),
                        ({"en": "Have you had sexual activity recently?", "hi": "क्या आपकी हाल में यौन गतिविधि हुई है?"}, 0.6, "yesno"),
                        ({"en": "Have you had urinary tract infections in the past?", "hi": "क्या आपको पहले भी मूत्र मार्ग का संक्रमण हुआ है?"}, 0.7, "yesno")
                    ],
                    {
                        "dysuria": 1.0, "urgency": 0.8, "frequency": 0.8,
                        "suprapubic pain": 0.7, "cloudy urine": 0.7, "hematuria": 0.8
                    }),

            # Intestinal Amoebiasis
            DiseaseNode("Intestinal Amoebiasis",
                    ["abdominal pain", "diarrhea", "blood in stools", "tenesmus", 
                        "weight loss", "fatigue", "abdominal_pain_diarrhea", "abdominalpain_diarrhea_vomiting"],
                    [
                        ({"en": "Is there blood or mucus in the stools?", "hi": "क्या मल में खून या बलगम आता है?"}, 0.9, "yesno"),
                        ({"en": "Have you traveled to an area where amoebiasis is common?", "hi": "क्या आप ऐसे क्षेत्र में गए हैं जहाँ अमीबियासिस आम है?"}, 0.8, "yesno"),
                        ({"en": "Have you had symptoms of a liver abscess (fever with right upper abdominal pain)?", "hi": "क्या आपको लिवर एब्सेस के लक्षण हुए हैं—बुखार के साथ दाहिने ऊपरी पेट में दर्द?"}, 0.7, "yesno")
                    ],
                    {
                        "abdominal pain": 0.8, "diarrhea": 0.9, "blood in stools": 1.0,
                        "tenesmus": 0.7, "weight loss": 0.7, "fatigue": 0.5,
                        "abdominal_pain_diarrhea": 0.8, "abdominalpain_diarrhea_vomiting": 0.8
                    }),

            # Gonorrhea in Men
            DiseaseNode("Gonorrhea in Men",
                    ["urethral discharge", "dysuria", "testicular pain", "erythema meatus"],
                    [
                        ({"en": "Is there pus-like discharge from the urethra?", "hi": "क्या मूत्रमार्ग से मवाद जैसा स्राव हो रहा है?"}, 0.9, "yesno"),
                        ({"en": "Have you had unprotected sexual intercourse recently?", "hi": "क्या आपका हाल में असुरक्षित यौन संबंध हुआ है?"}, 0.8, "yesno"),
                        ({"en": "Does your partner have any symptoms?", "hi": "क्या आपके साथी में कोई लक्षण हैं?"}, 0.7, "yesno")
                    ],
                    {
                        "urethral discharge": 1.0, "dysuria": 0.8, 
                        "testicular pain": 0.7, "erythema meatus": 0.7
                    }),

            # Gonorrhea in Women
            DiseaseNode("Gonorrhea in Women",
                    ["vaginal discharge", "dysuria", "pelvic pain", "intermenstrual bleeding"],
                    [
                        ({"en": "Is the vaginal discharge thick and pus-like (mucopurulent)?", "hi": "क्या योनि स्राव गाढ़ा और मवाद जैसा है?"}, 0.9, "yesno"),
                        ({"en": "Do you feel pain when the cervix is moved during examination (cervical motion tenderness)?", "hi": "जाँच के दौरान गर्भाशय ग्रीवा हिलाने पर क्या दर्द होता है?"}, 0.8, "yesno"),
                        (({"en": "Do you have fever along with these symptoms?", "hi": "क्या इन लक्षणों के साथ आपको बुखार भी है?"}), 0.6, "yesno")
                    ],
                    {
                        "vaginal discharge": 1.0, "dysuria": 0.8, 
                        "pelvic pain": 0.7, "intermenstrual bleeding": 0.7
                    }),

            # Oral Candidiasis
            DiseaseNode("Oral Candidiasis",
                    ["white oral plaques", "burning mouth", "loss of taste", "furry feeling"],
                    [
                        ({"en": "Do you have any condition that suppresses immunity (e.g., long-term steroids, chemotherapy)?",
                        "hi": "क्या आपको ऐसी स्थिति है जिससे प्रतिरक्षा कम होती है, जैसे लंबे समय तक स्टेरॉयड या कीमोथेरेपी?"}, 0.8, "yesno"),
                        ({"en": "Have you recently taken antibiotics?", "hi": "क्या आपने हाल में एंटीबायोटिक दवाएँ ली हैं?"}, 0.8, "yesno"),
                        ({"en": "Do you have diabetes?", "hi": "क्या आपको मधुमेह है?"}, 0.7, "yesno")
                    ],
                    {
                        "white oral plaques": 1.0, "burning mouth": 0.9, 
                        "loss of taste": 0.8, "furry feeling": 0.8
                    }),

            # Upper Respiratory Tract Infections
            DiseaseNode("Upper Respiratory Tract Infections",
                    ["fever", "headache", "fatigue", "sneezing", "fever_headache", "fever_fatigue"],
                    [
                        ({"en": "Do you have hoarseness or a rough voice?", "hi": "क्या आपकी आवाज़ बैठी हुई या भारी हो गई है?"}, 0.8, "yesno"),
                        ({"en": "Do you have ear pain?", "hi": "क्या आपको कान में दर्द है?"}, 0.8, "yesno"),
                        ({"en": "Do you have nasal blockage or a runny nose?", "hi": "क्या आपकी नाक बंद रहती है या नाक से पानी बहता है?"}, 0.7, "yesno")
                    ],
                    {
                        "sneezing": 1.0, "fever": 0.9, 
                        "headache": 0.8, "fatigue": 0.8,
                        "fever_headache": 0.7, "fever_fatigue": 0.7
                    }),

            # Mental Health Disorders
            DiseaseNode("Mental Health Disorders",
                    ["anxiety", "depression", "sleeping disorder","bipolar affective disorders", "schizophrenia", "dementia"],
                    [
                        ({"en": "Do you have difficulty with sleep (falling asleep or staying asleep)?",
                        "hi": "क्या आपको नींद आने या नींद बनाए रखने में कठिनाई होती है?"}, 0.8, "yesno"),
                        ({"en": "Have you been feeling depressed for most days?",
                        "hi": "क्या आप अधिकांश दिनों में उदास या अवसादग्रस्त महसूस करते हैं?"}, 0.8, "yesno"),
                        ({"en": "Do you have a history of using alcohol, tobacco, or other substances?",
                        "hi": "क्या आपके पास शराब, तंबाकू या अन्य नशीले पदार्थों के उपयोग का इतिहास है?"}, 0.7, "yesno")
                    ],
                    {
                        "dementia": 1.0, "depression": 0.9, "schizophrenia": 1.0,
                        "anxiety": 0.8, "bipolar affective disorders": 1.0, "sleeping disorder":0.9
                    }),
        ]
    
    def get_display_text(self, q_text, lang="both"):
        """
        Normalize question text for display.
        - If dict: return multilingual text (defaults to EN + HI)
        - If str: return directly
        """
        if isinstance(q_text, dict):
            normalized = _normalize_question_texts(q_text)
            if lang == "both":
                preferred_order = ["en", "hi"]
                combined = []
                for key in preferred_order + [l for l in SUPPORTED_LANGUAGES if l not in preferred_order]:
                    text = normalized.get(key, "")
                    if text and text not in combined:
                        combined.append(text)
                return " / ".join(combined)

            if lang in SUPPORTED_LANGUAGES:
                return normalized.get(lang, normalized.get("en", ""))

            return next((normalized.get(key, "") for key in SUPPORTED_LANGUAGES if normalized.get(key)), "")

        return _coerce_question_text(q_text)

    def calculate_symptom_confidence(self, symptom_name: str, responses: list) -> float:
        """
        Calculate confidence score for a symptom based on yes/no responses only.
        responses: list of tuples (question, answer, weight)
        """
        # Filter only boolean answers
        yn_responses = [(q, a, w) for q, a, w in responses if isinstance(a, bool)]
        total_weight = sum(weight for _, _, weight in yn_responses)
        if total_weight == 0:
            return 0.0

        positive_weight = sum(weight for _, ans, weight in yn_responses if ans)
        return positive_weight / total_weight
    
    ### hybrid symptom triggering
    def check_and_trigger_hybrids(self, initial_symptoms):
        """Check if any hybrid nodes should be triggered"""
        self.triggered_hybrids.clear()

        initial_set = {symptom.lower() for symptom in initial_symptoms}
        matched_hybrids = []
        for hybrid_name, hybrid_node in self.hybrid_symptom_nodes.items():
            if hybrid_node.component_symptoms_normalized.issubset(initial_set):
                matched_hybrids.append((hybrid_name, hybrid_node))

        if not matched_hybrids:
            return []

        maximal_hybrids = []
        for name, node in matched_hybrids:
            if not any(
                node.component_symptoms_normalized
                < other_node.component_symptoms_normalized
                for other_name, other_node in matched_hybrids
                if other_name != name
            ):
                maximal_hybrids.append((name, node))

        # Prioritize hybrids with more component symptoms, then higher weight, then name
        prioritized = sorted(
            maximal_hybrids,
            key=lambda item: (
                -len(item[1].component_symptoms),
                -item[1].weight,
                item[0],
            ),
        )

        for hybrid_name, _ in prioritized:
            self.triggered_hybrids.add(hybrid_name)
            print(f"\n🔗 Detected symptom combination: {hybrid_name}")

        return [name for name, _ in prioritized]

    def ask_dynamic_questions(self, symptom_name: str, max_questions: int = 2) -> float:
        """Ask dynamic questions for a symptom, ensuring child questions are always asked."""
        if symptom_name in self.symptom_confidences:
            return self.symptom_confidences[symptom_name]

        # Handle hybrids
        normalized_symptom = symptom_name.lower()
        for hybrid_name in self.triggered_hybrids:
            hybrid_node = self.hybrid_symptom_nodes[hybrid_name]
            if normalized_symptom in hybrid_node.component_symptoms_normalized:
                print(f"Skipping {symptom_name} - handled by {hybrid_name}")
                hybrid_confidence = self.symptom_confidences.get(hybrid_name, 1.0)
                self.symptom_confidences[symptom_name] = hybrid_confidence * 0.9
                return self.symptom_confidences[symptom_name]

        root = self.symptom_tree_roots.get(symptom_name)
        self.symptom_details.setdefault(symptom_name, {})

        if not root:
            ans = input(f"Do you have {symptom_name}? (yes/no): ").strip().lower()
            is_positive = ans.startswith('y')
            self.symptom_details[symptom_name]["generic_presence"] = is_positive
            confidence = 1.0 if is_positive else 0.0
            self.symptom_confidences[symptom_name] = confidence
            return confidence

        print(f"\n--- Evaluating {symptom_name} ---")

        questions = sorted(root.clarifying_questions, key=lambda x: x[1], reverse=True)
        responses = []

        for question in questions[:max_questions]:
            if len(question) == 2:
                q_text, weight = question
                q_type = "yesno"
            else:
                q_text, weight, q_type = question

            display_text = self.get_display_text(q_text)
            key_text = display_text  # always store string key

            if key_text in self.symptom_details[symptom_name]:
                prev_ans = self.symptom_details[symptom_name][key_text]
                responses.append((q_text, prev_ans, weight))
                continue

            if q_type == "yesno":
                ans = input(f"{display_text} (yes/no): ").strip().lower()
                is_positive = ans.startswith('y')
                responses.append((q_text, is_positive, weight))
                self.symptom_details[symptom_name][key_text] = is_positive

            elif q_type == "number":
                ans = input(f"{display_text} (enter number): ").strip()
                try:
                    num_value = float(ans)
                except ValueError:
                    num_value = None
                responses.append((q_text, num_value, weight))
                self.symptom_details[symptom_name][key_text] = num_value

            elif q_type == "text":
                ans = input(f"{display_text}: ").strip()
                responses.append((q_text, ans, weight))
                self.symptom_details[symptom_name][key_text] = ans

            elif q_type == "choice":
                ans = input(f"{display_text}: ").strip().lower()
                responses.append((q_text, ans, weight))
                self.symptom_details[symptom_name][key_text] = ans

        confidence = self.calculate_symptom_confidence(symptom_name, responses)
        self.symptom_confidences[symptom_name] = confidence

        if root.children:
            print(f"\nExploring related symptoms for {symptom_name}:")
            sorted_children = sorted(root.children, key=lambda x: x.weight, reverse=True)
            for child in sorted_children[:min(len(sorted_children), 3)]:
                skip_child = False
                child_name_normalized = child.name.lower()
                for hybrid_name in self.triggered_hybrids:
                    hybrid_node = self.hybrid_symptom_nodes[hybrid_name]
                    if (
                        child_name_normalized
                        in hybrid_node.component_symptoms_normalized
                    ):
                        print(f"\n  → Skipping {child.name} - handled by {hybrid_name}")
                        skip_child = True
                        break
                if skip_child:
                    continue

                print(f"\n  → Checking: {child.name}")
                child_confidence = self.ask_child_questions(child, max_questions=1)
                self.symptom_confidences[child.name] = child_confidence
                if child_confidence >= 0.7:
                    self.confirmed_symptoms.add(child.name)
                    print(f" ✓ {child.name} confirmed ({child_confidence:.2f})")
                else:
                    print(f"✗ {child.name} not confirmed ({child_confidence:.2f})")

        return confidence
    

    def ask_hybrid_questions(self, hybrid_name: str) -> float:
        """Ask questions specific to hybrid symptom combinations"""
        hybrid_node = self.hybrid_symptom_nodes[hybrid_name]
        print(f"\n--- Evaluating {hybrid_name} combination ---")

        detail_target = self.symptom_details.setdefault(hybrid_name, {})
        if hybrid_name == self.gi_hybrid_key:
            self.reset_gi_followup_state()
            self.run_gi_dynamic_followup(detail_target)
        responses = []
        for question in hybrid_node.hybrid_questions:
            q_text, weight, q_type = question
            display_text = self.get_display_text(q_text)
            key_text = display_text

            if q_type == "yesno":
                ans = input(f"{display_text} (yes/no): ").strip().lower()
                is_positive = ans.startswith('y')
                responses.append((q_text, is_positive, weight))
                detail_target[key_text] = is_positive

        confidence = self.calculate_symptom_confidence(hybrid_name, responses)
        self.symptom_confidences[hybrid_name] = confidence

        if confidence >= 0.7:
            for component in hybrid_node.component_symptoms:
                self.confirmed_symptoms.add(component)
                self.symptom_confidences[component] = confidence * 0.9

        return confidence

    def ask_child_questions(self, child_node: SymptomNode, max_questions: int = 1) -> float:
        """Ask questions specifically for child nodes with limited questions"""
        responses = []
        self.symptom_details.setdefault(child_node.name, {})

        questions = sorted(child_node.clarifying_questions, key=lambda x: x[1], reverse=True)
        for question in questions[:max_questions]:
            if len(question) == 2:
                q_text, weight = question
                q_type = "yesno"
            else:
                q_text, weight, q_type = question

            display_text = self.get_display_text(q_text)
            key_text = display_text

            if q_type == "yesno":
                ans = input(f"    {display_text} (yes/no): ").strip().lower()
                is_positive = ans.startswith('y')
                responses.append((q_text, is_positive, weight))
                self.symptom_details[child_node.name][key_text] = is_positive

            elif q_type == "number":
                ans = input(f"    {display_text} (enter number): ").strip()
                try:
                    num_value = float(ans)
                except ValueError:
                    num_value = None
                responses.append((q_text, num_value, weight))
                self.symptom_details[child_node.name][key_text] = num_value

            elif q_type == "text":
                ans = input(f"    {display_text}: ").strip()
                responses.append((q_text, ans, weight))
                self.symptom_details[child_node.name][key_text] = ans

            elif q_type == "choice":
                ans = input(f"    {display_text}: ").strip().lower()
                responses.append((q_text, ans, weight))
                self.symptom_details[child_node.name][key_text] = ans

        return self.calculate_symptom_confidence(child_node.name, responses)

    def calculate_disease_score(self, disease: DiseaseNode) -> float:
        """Calculate likelihood score for a disease based on confirmed symptoms including hybrids"""
        
        total_possible_score = sum(disease.symptom_weights.get(s, 1.0) for s in disease.required_symptoms)
        actual_score = 0.0
        
        for symptom in disease.required_symptoms:
            if symptom in self.confirmed_symptoms:
                symptom_weight = disease.symptom_weights.get(symptom, 1.0)
                symptom_confidence = self.symptom_confidences.get(symptom, 0.0)
                actual_score += symptom_weight * symptom_confidence
            
            # NEW: Check if symptom is a hybrid and if its components are confirmed
            elif symptom in self.hybrid_symptom_nodes:
                hybrid_node = self.hybrid_symptom_nodes[symptom]
                # Check if all components of hybrid are confirmed individually
                if hybrid_node.component_symptoms.issubset(self.confirmed_symptoms):
                    # Calculate average confidence of components
                    component_confidences = [
                        self.symptom_confidences.get(comp, 0.0) 
                        for comp in hybrid_node.component_symptoms
                    ]
                    avg_confidence = sum(component_confidences) / len(component_confidences)
                    symptom_weight = disease.symptom_weights.get(symptom, 1.0)
                    actual_score += symptom_weight * avg_confidence
        
        return actual_score / total_possible_score if total_possible_score > 0 else 0.0

    def ask_disease_specific_questions(self, disease: DiseaseNode) -> float:
        """Ask disease-specific questions with multiple answer types."""
        print(f"\n--- Evaluating for {disease.name} ---")
        print("Answer these key diagnostic questions:")

        responses = []
        self.symptom_details.setdefault(f"{disease.name}_diagnostic", {})

        top_questions = sorted(disease.diagnostic_questions, key=lambda x: x[1], reverse=True)[:3]

        for q_text, weight, q_type in top_questions:
            display_text = self.get_display_text(q_text)
            key_text = display_text

            if q_type == "yesno":
                ans = input(f"{display_text} (yes/no): ").strip().lower()
                is_positive = ans.startswith('y')
                responses.append((q_text, is_positive, weight))
                self.symptom_details[f"{disease.name}_diagnostic"][key_text] = is_positive

            elif q_type == "number":
                ans = input(f"{display_text} (enter number): ").strip()
                try:
                    num_value = float(ans)
                except ValueError:
                    num_value = None
                responses.append((q_text, num_value, weight))
                self.symptom_details[f"{disease.name}_diagnostic"][key_text] = num_value

            elif q_type == "text":
                ans = input(f"{display_text}: ").strip()
                responses.append((q_text, ans, weight))
                self.symptom_details[f"{disease.name}_diagnostic"][key_text] = ans

            elif q_type == "choice":
                ans = input(f"{display_text}: ").strip().lower()
                responses.append((q_text, ans, weight))
                self.symptom_details[f"{disease.name}_diagnostic"][key_text] = ans

        return self.calculate_symptom_confidence(f"{disease.name}_diagnostic", responses)

    def intelligent_disease_evaluation(self):
        """Evaluate the top 2 most likely diseases based on confirmed symptoms."""

        # Step 1: Score all diseases and pick top 2
        disease_candidates = [
            (disease, self.calculate_disease_score(disease))
            for disease in self.diseases
            if self.calculate_disease_score(disease) > 0.1
        ]
        disease_candidates.sort(key=lambda x: x[1], reverse=True)
        top_diseases = disease_candidates[:2]

        print(f"\nEvaluating {len(top_diseases)} most likely conditions...")

        # Step 2: Evaluate each candidate
        for disease, initial_score in top_diseases:
            print(f"\n{'=' * 50}")
            print(f"Evaluating: {disease.name} (Initial match: {initial_score:.2f})")

            # Ask about missing high-weight symptoms (limit to 3)
            missing_critical = [
                (symptom, disease.symptom_weights.get(symptom, 1.0))
                for symptom in disease.required_symptoms
                if symptom not in self.confirmed_symptoms
                and disease.symptom_weights.get(symptom, 1.0) >= 0.8
            ]
            for symptom, _ in sorted(missing_critical, key=lambda x: x[1], reverse=True)[:3]:
                print(f"\nChecking critical symptom: {symptom}")
                confidence = self.ask_dynamic_questions(symptom, max_questions=2)
                if confidence >= 0.6:
                    self.confirmed_symptoms.add(symptom)
                    print(f"✓ {symptom} confirmed ({confidence:.2f})")
                else:
                    print(f"✗ {symptom} not confirmed ({confidence:.2f})")

            # Step 3: Recalculate and optionally run disease-specific questions
            updated_score = self.calculate_disease_score(disease)
            if updated_score >= disease.confidence_threshold:
                diag_conf = self.ask_disease_specific_questions(disease)
                final_score = (updated_score + diag_conf) / 2
                self.disease_scores[disease.name] = final_score
                if final_score >= 0.7:
                    return disease, final_score
            else:
                print(f"Skipping {disease.name} - insufficient match ({updated_score:.2f})")

        # Step 4: Return best match even if under threshold
        if self.disease_scores:
            best_name, best_score = max(self.disease_scores.items(), key=lambda x: x[1])
            best_disease = next(d for d in self.diseases if d.name == best_name)
            return best_disease, best_score

        return None, 0.0

    def run_enhanced_diagnostic(self):
        """Run the full enhanced diagnostic workflow."""
        self.build_symptom_tree()
        self.build_hybrid_nodes()  # NEW: Build hybrid nodes
        self.build_disease_knowledge()

        print("🏥 Enhanced Medical Diagnostic System")
        print("=" * 50)
        user_input = input("\nEnter your main symptoms (comma-separated): ").strip()
        if not user_input:
            print("No symptoms provided. Exiting...")
            return

        initial_symptoms = [s.strip().lower() for s in user_input.split(",") if s.strip()]
        print(f"\n📋 Initial symptoms: {', '.join(initial_symptoms)}")

        # NEW: Check for hybrid triggers BEFORE individual symptom evaluation
        triggered_hybrids = self.check_and_trigger_hybrids(initial_symptoms)
        
        # Symptom evaluation phase
        print("\n" + "=" * 50)
        print("SYMPTOM EVALUATION PHASE")
        print("=" * 50)
        
        # NEW: Handle hybrid symptoms first if triggered
        handled_components = set()
        if triggered_hybrids:
            for hybrid_name in triggered_hybrids:
                print(f"\n🔗 Processing symptom combination: {hybrid_name}")
                hybrid_confidence = self.ask_hybrid_questions(hybrid_name)
                if hybrid_confidence >= 0.5:
                    self.confirmed_symptoms.add(hybrid_name)
                    print(f"✓ {hybrid_name} confirmed ({hybrid_confidence:.2f})")

                    # Add component symptoms to confirmed list
                    hybrid_node = self.hybrid_symptom_nodes[hybrid_name]
                    for component in hybrid_node.component_symptoms:
                        self.confirmed_symptoms.add(component)
                        handled_components.add(component.lower())
                        print(
                            f"✓ {component} confirmed via {hybrid_name} ({hybrid_confidence * 0.9:.2f})"
                        )
                else:
                    print(f"✗ {hybrid_name} not confirmed ({hybrid_confidence:.2f})")
                handled_components.update(
                    {
                        symptom.lower()
                        for symptom in self.hybrid_symptom_nodes[hybrid_name].component_symptoms
                    }
                )

            # Get remaining symptoms not handled by any triggered hybrid
            remaining_symptoms = [
                s for s in initial_symptoms if s not in handled_components
            ]
        else:
            remaining_symptoms = initial_symptoms

        # Process remaining individual symptoms
        for symptom in remaining_symptoms:
            confidence = self.ask_dynamic_questions(symptom, max_questions=2)
            if confidence >= 0.5:
                self.confirmed_symptoms.add(symptom)
                print(f"✓ {symptom} confirmed ({confidence:.2f})")
            else:
                print(f"✗ {symptom} not confirmed ({confidence:.2f})")

        if not self.confirmed_symptoms:
            print("\n⚠ No symptoms confirmed. Please consult a healthcare provider.")
            return

        print(f"\n✅ Confirmed symptoms: {', '.join(self.confirmed_symptoms)}")

        # Disease evaluation phase
        print("\n" + "=" * 50)
        print("DISEASE EVALUATION PHASE (Top 2 most likely)")
        print("=" * 50)
        top_disease, confidence = self.intelligent_disease_evaluation()

        # Results
        print("\n" + "=" * 50)
        print("DIAGNOSTIC RESULTS")
        print("=" * 50)
        if top_disease and confidence >= 0.5:
            print(f"🎯 Most Likely Diagnosis: {top_disease.name}")
            print(f"📊 Confidence Score: {confidence:.2f}")
            print("\n📋 Key Supporting Symptoms:")
            for symptom in top_disease.required_symptoms:
                if symptom in self.confirmed_symptoms:
                    conf = self.symptom_confidences.get(symptom, 0.0)
                    print(f"  ✓ {symptom} ({conf:.2f})")
            
            # NEW: Show if any hybrid symptoms contributed
            if triggered_hybrids:
                confirmed_hybrids = [
                    name
                    for name in triggered_hybrids
                    if name in self.confirmed_symptoms
                ]
                if confirmed_hybrids:
                    print(f"\n🔗 Hybrid Symptom Pattern:")
                    for name in confirmed_hybrids:
                        print(
                            f"  ✓ {name} ({self.symptom_confidences.get(name, 0.0):.2f})"
                        )
                
            print("\n🏥 Recommendations:")
            print("  • Seek medical consultation for confirmation")
            if confidence < 0.7:
                print("  • Further tests may be needed for definitive diagnosis")
        else:
            print("❓ No clear diagnosis reached from top likely conditions")
            if self.disease_scores:
                print("\n📊 Considered conditions:")
                for name, score in sorted(self.disease_scores.items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"  • {name}: {score:.2f}")
            print("\n🏥 Recommendations:")
            print("  • Consult a healthcare provider for proper evaluation")
            print("  • Consider additional tests or specialist referral")

        # Summary of all collected information
        print("\n" + "=" * 30)
        print("SUMMARY OF FINDINGS")
        print("=" * 30)
        print(f"Confirmed Symptoms: {len(self.confirmed_symptoms)}")
        
        # NEW: Separate hybrid and individual symptoms in summary
        individual_symptoms = []
        hybrid_symptoms = []
        
        for symptom in self.confirmed_symptoms:
            if symptom in self.hybrid_symptom_nodes:
                hybrid_symptoms.append(symptom)
            else:
                individual_symptoms.append(symptom)
        
        if hybrid_symptoms:
            print("🔗 Hybrid Symptom Patterns:")
            for symptom in hybrid_symptoms:
                conf = self.symptom_confidences.get(symptom, 0.0)
                print(f"  • {symptom}: {conf:.2f}")
        
        if individual_symptoms:
            print("Individual Symptoms:")
            for symptom in individual_symptoms:
                conf = self.symptom_confidences.get(symptom, 0.0)
                print(f"  • {symptom}: {conf:.2f}")
        
        print(f"\nEvaluated Diseases: {len(self.disease_scores)}")
        for name, score in sorted(self.disease_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {name}: {score:.2f}")

# Example usage and testing
if __name__ == "__main__":
    engine = DiagnosticEngine()
    engine.run_enhanced_diagnostic()

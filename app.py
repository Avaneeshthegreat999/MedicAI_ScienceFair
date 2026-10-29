import os
import secrets
from flask import Flask, render_template, request, jsonify, session
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM
import shutil

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# ── Config ─────────────────────────────────────────────────────────────────
UPLOAD_FOLDER = "uploads"
CHROMA_DIR    = "chroma_db"
OLLAMA_MODEL  = "llama3.2:1b"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_DIR,    exist_ok=True)

# ── Built-in Medical Knowledge ─────────────────────────────────────────────
BUILTIN_KNOWLEDGE = [
    Document(page_content="""
    Diabetes Mellitus: Chronic metabolic disease with elevated blood glucose.
    Type 1: Autoimmune beta cell destruction. Requires insulin.
    Type 2: Insulin resistance. Managed with metformin, lifestyle changes, insulin.
    Symptoms: Polyuria, polydipsia, polyphagia, fatigue, blurred vision, slow healing.
    Diagnosis: Fasting glucose >=126 mg/dL, HbA1c >=6.5%.
    Complications: Retinopathy, nephropathy, neuropathy, cardiovascular disease.
    """, metadata={"source": "Built-in Knowledge Base", "topic": "Diabetes"}),

    Document(page_content="""
    Hypertension: Persistent high blood pressure >=130/80 mmHg.
    Often asymptomatic. Severe: headache, shortness of breath, nosebleeds.
    Risk factors: Obesity, high sodium, inactivity, smoking, alcohol, family history.
    Treatment: ACE inhibitors, ARBs, calcium channel blockers, diuretics, beta-blockers.
    Complications: Stroke, heart attack, heart failure, kidney disease.
    """, metadata={"source": "Built-in Knowledge Base", "topic": "Hypertension"}),

    Document(page_content="""
    Myocardial Infarction (Heart Attack): Necrosis of heart muscle due to ischemia.
    Symptoms: Chest pain radiating to arm/jaw, shortness of breath, sweating, nausea.
    Women: Atypical symptoms — fatigue, back pain, jaw pain.
    Treatment: MONA (Morphine, Oxygen, Nitrates, Aspirin), PCI, thrombolytics.
    Emergency: Call emergency services immediately. Door-to-balloon time < 90 minutes.
    """, metadata={"source": "Built-in Knowledge Base", "topic": "Heart Attack"}),

    Document(page_content="""
    Pneumonia: Lung infection causing inflammation of air sacs.
    Types: Bacterial (Streptococcus), Viral (influenza, COVID-19), Fungal, Aspiration.
    Symptoms: Cough with phlegm, fever, chills, shortness of breath, chest pain.
    Treatment: Bacterial — antibiotics. Viral — supportive care.
    Prevention: Pneumococcal vaccine, influenza vaccine, hand hygiene.
    """, metadata={"source": "Built-in Knowledge Base", "topic": "Pneumonia"}),

    Document(page_content="""
    Asthma: Chronic airway inflammation causing wheezing, breathlessness, chest tightness.
    Triggers: Allergens, exercise, cold air, infections, stress, smoke.
    Rescue inhaler: Salbutamol/Albuterol (SABA).
    Long-term: Inhaled corticosteroids (ICS), LABA, leukotriene modifiers.
    Diagnosis: Spirometry FEV1/FVC < 0.70, bronchodilator reversibility.
    """, metadata={"source": "Built-in Knowledge Base", "topic": "Asthma"}),

    Document(page_content="""
    Stroke: Sudden interruption of brain blood supply.
    Types: Ischemic (87% — clot) and Hemorrhagic (13% — bleed).
    FAST: Face drooping, Arm weakness, Speech difficulty, Time to call emergency.
    Treatment Ischemic: tPA within 4.5 hours, mechanical thrombectomy.
    Risk factors: Hypertension, atrial fibrillation, diabetes, smoking.
    """, metadata={"source": "Built-in Knowledge Base", "topic": "Stroke"}),

    Document(page_content="""
    COVID-19: Disease caused by SARS-CoV-2.
    Symptoms: Fever, dry cough, fatigue, loss of taste/smell, shortness of breath.
    Severe: Difficulty breathing, chest pain, confusion, bluish lips.
    Treatment: Antivirals (Paxlovid, remdesivir), corticosteroids, oxygen therapy.
    Long COVID: Persistent symptoms weeks/months after acute infection.
    """, metadata={"source": "Built-in Knowledge Base", "topic": "COVID-19"}),

    Document(page_content="""
    Depression: Persistent sadness, loss of interest, fatigue, sleep disturbances.
    Diagnosis: DSM-5 — 5+ symptoms for 2+ weeks causing significant distress.
    Treatment: CBT psychotherapy, SSRIs (fluoxetine, sertraline), SNRIs, combination.
    Suicide risk: Always take seriously. Refer to mental health professional immediately.
    """, metadata={"source": "Built-in Knowledge Base", "topic": "Depression"}),

    Document(page_content="""
    Common Medications:
    Antibiotics: Amoxicillin, Azithromycin, Ciprofloxacin, Metronidazole.
    Pain: Paracetamol, Ibuprofen, Aspirin, Morphine.
    Cardiovascular: Atenolol, Amlodipine, Lisinopril, Atorvastatin.
    Diabetes: Metformin, Insulin, Glipizide, Empagliflozin.
    Always consult a doctor before taking any medication.
    """, metadata={"source": "Built-in Knowledge Base", "topic": "Medications"}),

    Document(page_content="""
    Emergency Symptoms — Call emergency services immediately:
    - Chest pain or pressure
    - Difficulty breathing
    - Sudden severe headache
    - Face drooping, arm weakness, speech difficulty (stroke)
    - Loss of consciousness
    - Severe allergic reaction (anaphylaxis)
    - Uncontrolled bleeding
    - Seizures
    - High fever >103F with stiff neck, rash, or confusion
    """, metadata={"source": "Built-in Knowledge Base", "topic": "Emergency Guidelines"}),
]

# ── Medical keywords for filtering ────────────────────────────────────────
MEDICAL_KEYWORDS = [
    "symptom","disease","treatment","medicine","medication","drug","dose",
    "doctor","hospital","diagnosis","medical","health","pain","fever",
    "infection","virus","bacteria","cancer","blood","heart","lung","brain",
    "kidney","liver","diabetes","hypertension","asthma","stroke","covid",
    "vaccine","surgery","therapy","prescription","allergy","chronic","acute",
    "patient","clinical","anatomy","physiology","pharmacy","nurse","physician",
    "injury","wound","bone","muscle","nerve","skin","rash","headache",
    "nausea","vomiting","diarrhea","fatigue","cough","breath","chest",
    "anxiety","depression","mental","psychiatric","nutrition","diet","vitamin",
    "supplement","exercise","weight","obesity","pregnancy","emergency",
    "cholesterol","glucose","pressure","mri","ct","xray","scan","biopsy",
    "what is","how to","can i","should i","is it","why do","when to",
    "how does","what causes","how long","what are","signs of","cure for"
]

def is_medical_question(q: str) -> bool:
    q = q.lower()
    return any(kw in q for kw in MEDICAL_KEYWORDS)

# ── Embeddings & Vector Store ──────────────────────────────────────────────
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

def get_vectorstore():
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    vs = Chroma.from_documents(
        documents=BUILTIN_KNOWLEDGE,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vs.persist()
    return vs

vectorstore = get_vectorstore()

# ── LLM ───────────────────────────────────────────────────────────────────
llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.3)

# ── Prompts ───────────────────────────────────────────────────────────────
MEDICAL_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are MediBot, an expert AI medical assistant with deep knowledge
of diseases, symptoms, medications, and treatments.

YOUR BEHAVIOR:
- Give DETAILED, SPECIFIC medical information directly
- For common conditions (cold, cough, fever, infection, headache, acidity etc.)
  provide SPECIFIC medications, dosages, and home remedies directly
- For serious conditions (chest pain, stroke, cancer etc.) provide full information
  AND advise seeing a doctor
- NEVER refuse to give medication names for common conditions
- NEVER just say "consult a doctor" without first giving proper information
- Give medications with their common brand names AND generic names
- Include dosage, frequency, and duration when recommending medications
- Mention side effects of medications when relevant
- Only say "consult a doctor" at the END as an additional note, not as the main answer
- Be like a knowledgeable doctor friend who gives real, useful information

EXAMPLES OF GOOD RESPONSES:
Q: I have a cough
A: For a dry cough, Dextromethorphan (brand: Robitussin) 10-20mg every 4-6 hours
   helps suppress it. For wet/productive cough, Guaifenesin (brand: Mucinex)
   400-600mg every 4 hours helps loosen mucus. You can also try:
   - Honey + warm water (natural cough suppressant)
   - Steam inhalation with Vicks VapoRub
   - Lozenges like Strepsils or Halls
   If cough persists more than 2 weeks or you have blood in sputum, see a doctor.

Q: I have a fever of 100F
A: Take Paracetamol/Acetaminophen (brand: Crocin, Dolo 650) 500-650mg every
   6 hours. You can also take Ibuprofen (brand: Brufen, Advil) 400mg every 8 hours
   with food. Stay hydrated, rest, and use a cold compress on your forehead.
   If fever exceeds 103F or lasts more than 3 days, consult a doctor.

Context from medical documents:
{context}

Chat History:
{chat_history}

Question: {question}

MediBot (give specific, detailed medical information first, then any caution at end):"""
)

SYMPTOM_PROMPT = PromptTemplate(
    input_variables=["symptoms"],
    template="""You are MediBot, an expert medical symptom assessment assistant.

Patient reports: {symptoms}

Give a DETAILED structured assessment:

1. MOST LIKELY CONDITIONS (explain each with reasoning)
2. URGENCY LEVEL (Emergency / Urgent / Can manage at home)
3. MEDICATIONS & TREATMENTS
   - Specific drug names (generic + brand)
   - Dosages and frequency
   - Home remedies that help
4. WHAT TO AVOID
5. RED FLAGS (symptoms that mean you must see a doctor immediately)
6. WHEN TO SEE A DOCTOR (be specific — not just "consult a doctor")

Be specific and helpful like a knowledgeable doctor friend.
Only recommend emergency care for genuinely serious symptoms."""
)

# ── Session memory ────────────────────────────────────────────────────────
session_memories = {}

def get_memory(sid):
    if sid not in session_memories:
        session_memories[sid] = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        )
    return session_memories[sid]

def get_chain(sid):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=get_memory(sid),
        combine_docs_chain_kwargs={"prompt": MEDICAL_PROMPT},
        return_source_documents=True,
        verbose=True
    )

# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if "session_id" not in session:
        session["session_id"] = secrets.token_hex(16)
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    mode = data.get("mode", "chat")
    sid = session.get("session_id", "default")

    if not msg:
        return jsonify({"error": "Please enter a message."}), 400

    try:
        history = ""
        if sid in session_memories:
            messages = session_memories[sid].chat_memory.messages[-10:]
            for m in messages:
                role = "You" if m.type == "human" else "MedicAI"
                history += f"{role}: {m.content}\n"

        if mode == "symptom":
            prompt = f"""You are MedicAI, an informational health assistant.

Assess the symptoms cautiously. Explain possible causes without claiming
a definite diagnosis. Mention warning signs and when professional care
is appropriate. Do not prescribe personalised medication doses.

Conversation:
{history}

Reported symptoms:
{msg}

Respond clearly with:
1. Possible explanations
2. What the person can do safely
3. Warning signs
4. When to seek medical care

Remind the user that this is informational, not a medical diagnosis.
"""
            response = llm.invoke(prompt)
            sources = []

        elif is_medical_question(msg):
            docs = vectorstore.similarity_search(msg, k=4)
            context = "\n\n".join(d.page_content for d in docs)

            prompt = f"""You are MedicAI, a helpful medical information assistant.

Answer the user's health question in clear, understandable language.
Use the medical context when relevant, but do not invent facts or claim
that you have diagnosed the user. Do not give personalised prescriptions
or medication dosages. Explain important warning signs and recommend
professional medical care when appropriate.

Conversation history:
{history}

Reference information:
{context}

User's question:
{msg}

MedicAI's answer:
"""
            response = llm.invoke(prompt)

            sources = []
            seen = set()
            for doc in docs:
                src = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "")
                topic = doc.metadata.get("topic", "")
                key = f"{src}-{page}"

                if key not in seen:
                    seen.add(key)
                    sources.append({
                        "source": src,
                        "page": page,
                        "topic": topic,
                        "snippet": doc.page_content[:150].strip() + "..."
                    })

        else:
            prompt = f"""You are MedicAI, a friendly general-purpose AI assistant.
Answer everyday questions naturally, including greetings, casual chat,
science, technology, programming, education, and general knowledge.
Be helpful, clear, and conversational. Do not claim to be human.
For medical questions, provide general information rather than a diagnosis.

Conversation history:
{history}

User: {msg}
MedicAI:
"""
            response = llm.invoke(prompt)
            sources = []

        response = str(response).strip()

        if sid not in session_memories:
            session_memories[sid] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

        session_memories[sid].chat_memory.add_user_message(msg)
        session_memories[sid].chat_memory.add_ai_message(response)

        return jsonify({
            "response": response,
            "sources": sources,
            "mode": mode
        })

    except Exception as e:
        app.logger.exception("Chat request failed")
        return jsonify({"error": f"Chat error: {str(e)}"}), 500

    
@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if not file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a PDF file"}), 400
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        loader   = PyPDFLoader(filepath)
        pages    = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks   = splitter.split_documents(pages)
        vectorstore.add_documents(chunks)
        vectorstore.persist()
        return jsonify({
            "message": f"✅ '{filename}' uploaded! {len(chunks)} chunks added to knowledge base.",
            "chunks": len(chunks), "pages": len(pages)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/clear", methods=["POST"])
def clear_history():
    sid = session.get("session_id", "default")
    if sid in session_memories:
        del session_memories[sid]
    return jsonify({"message": "Chat history cleared."})


@app.route("/clear_kb", methods=["POST"])
def clear_knowledge_base():
    global vectorstore
    try:
        # Delete all entries from the collection
        collection = vectorstore._collection
        all_ids = collection.get()["ids"]

        if all_ids:
            collection.delete(ids=all_ids)

        # Re-add only built-in knowledge
        vectorstore.add_documents(BUILTIN_KNOWLEDGE)

        return jsonify({"message": f"✅ Cleared {len(all_ids)} chunks. Built-in knowledge restored."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  MediBot — Medical AI Chatbot")
    print("  Before running, start Ollama:")
    print("  > ollama serve")
    print("  > ollama pull llama3")
    print("="*50 + "\n")
    app.run(debug=True, port=5000)

import os
import json
import uuid
import time
import asyncio  # Ajout de cette ligne
from datetime import datetime
from io import BytesIO
from functools import lru_cache
from typing import List, Optional

import openai
import numpy as np
import PyPDF2
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, File, UploadFile, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache

load_dotenv()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    FastAPICache.init(InMemoryBackend())
    yield

app = FastAPI(lifespan=lifespan)



app.add_middleware(GZipMiddleware, minimum_size=1000)


app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
CANDIDATURES_DIR = os.path.join(UPLOAD_DIR, "candidatures")
OFFRES_DIR = os.path.join(UPLOAD_DIR, "offres")

os.makedirs(CANDIDATURES_DIR, exist_ok=True)
os.makedirs(OFFRES_DIR, exist_ok=True)


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
if torch.cuda.is_available():
    model = model.to('cuda')


openai.api_key = os.getenv("OPENAI_API_KEY")


class FileData(BaseModel):
    id: str
    original_filename: str
    filename: str
    content_type: str
    file_path: str
    contenu: str
    date_creation: datetime

class MatchRequest(BaseModel):
    cv_id: str
    job_id: str

class MatchResult(BaseModel):
    cv_id: str
    job_id: str
    score: float
    correspondances: List[str]
    non_correspondances: List[str]
    justification: str
    cv_summary: str
    job_summary: str

# Fonctions utilitaires
@lru_cache(maxsize=1000)
def get_embedding(text: str) -> np.ndarray:
    with torch.no_grad():
        return model.encode([text])[0]

def extract_text_from_pdf(file_content: bytes) -> str:
    pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

async def save_file(file: UploadFile, directory: str) -> FileData:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Seuls les fichiers PDF sont acceptés")

    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{file_id}{file_extension}"
    file_location = os.path.join(directory, unique_filename)

    try:
        file_content = await file.read()
        with open(file_location, "wb") as file_object:
            file_object.write(file_content)

        contenu = extract_text_from_pdf(file_content)

        file_data = FileData(
            id=file_id,
            original_filename=file.filename,
            filename=unique_filename,
            content_type=file.content_type,
            file_path=file_location,
            contenu=contenu,
            date_creation=datetime.now()
        )

        # Save metadata
        metadata_file = os.path.join(directory, f"{file_id}.json")
        with open(metadata_file, "w") as f:
            json.dump(file_data.dict(), f, default=str)

        return file_data
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du fichier: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la sauvegarde du fichier: {str(e)}")

async def rate_limited_api_call(func, *args, **kwargs):
    max_retries = 5
    for i in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except openai.error.RateLimitError:
            if i == max_retries - 1:
                raise
            wait_time = 2 ** i
            print(f"Rate limit atteint. Attente de {wait_time} secondes.")
            time.sleep(wait_time)

@cache(expire=3600)
async def extract_skills_and_info(text: str, is_cv: bool) -> str:
    role = "Tu es un expert en analyse de CV" if is_cv else "Tu es un expert en analyse d'offres d'emploi"
    try:
        response = await rate_limited_api_call(
            openai.ChatCompletion.acreate,
            model='gpt-4',
            messages=[
                {'role': 'system', 'content': f"{role}. Extrais les informations importantes et liste les compétences. Limitez-vous aux 10 éléments les plus importants pour chaque catégorie."},
                {'role': 'user', 'content': text[:3000]}
            ],
            max_tokens=500
        )
        return response.choices[0]['message']['content']
    except Exception as e:
        print(f"Erreur OpenAI lors de l'extraction des compétences: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse du contenu: {str(e)}")

@cache(expire=3600)
async def compare_cv_to_job(cv_text: str, job_text: str) -> dict:
    try:
        response = await rate_limited_api_call(
            openai.ChatCompletion.acreate,
            model='gpt-4',
            messages=[
                {'role': 'system', 'content': "Tu es un expert en recrutement. Compare ce CV à cette offre d'emploi. Donne un score de correspondance en pourcentage, liste les correspondances et non-correspondances, et justifie ton analyse. Fournis ta réponse au format JSON avec les clés 'score' (un nombre entre 0 et 100), 'correspondances', 'non_correspondances', et 'justification'."},
                {'role': 'user', 'content': f"CV : {cv_text[:1500]}\n\nOffre d'emploi : {job_text[:1500]}"}
            ],
            max_tokens=1000
        )
        result = json.loads(response.choices[0]['message']['content'])
        return validate_match_result(result)
    except json.JSONDecodeError as e:
        print(f"Erreur de décodage JSON: {str(e)}")
        print(f"Réponse brute: {response.choices[0]['message']['content']}")
        raise HTTPException(status_code=500, detail="Erreur lors du traitement de la réponse")
    except Exception as e:
        print(f"Erreur OpenAI lors de la comparaison CV-offre: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la comparaison CV-offre: {str(e)}")

def validate_match_result(result: dict) -> dict:
    required_keys = ['score', 'correspondances', 'non_correspondances', 'justification']
    if not all(key in result for key in required_keys):
        raise ValueError("La réponse ne contient pas toutes les clés requises")
    if not isinstance(result['score'], (int, float)) or not 0 <= result['score'] <= 100:
        raise ValueError("Le score doit être un nombre entre 0 et 100")
    return result

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload_cv/", response_model=FileData)
async def upload_cv(file: UploadFile = File(...)):
    return await save_file(file, CANDIDATURES_DIR)

@app.post("/offres_emploi/", response_model=FileData)
async def create_offre_emploi(file: UploadFile = File(...)):
    return await save_file(file, OFFRES_DIR)

@app.get("/candidatures/", response_model=List[FileData])
def read_candidatures(
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = Query(None, description="Rechercher par nom de fichier"),
    sort_by_date: bool = Query(False, description="Trier par date de création (le plus récent en premier)")
):
    return get_files(CANDIDATURES_DIR, skip, limit, search, sort_by_date)

@app.get("/offres_emploi/", response_model=List[FileData])
def read_offres_emploi(
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = Query(None, description="Rechercher dans le contenu ou le nom du fichier"),
    sort_by_date: bool = Query(False, description="Trier par date de création (le plus récent en premier)")
):
    return get_files(OFFRES_DIR, skip, limit, search, sort_by_date)

def get_files(directory: str, skip: int, limit: int, search: Optional[str], sort_by_date: bool) -> List[FileData]:
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            with open(os.path.join(directory, filename), "r") as f:
                file_data = json.load(f)
                if search is None or search.lower() in file_data["original_filename"].lower() or search.lower() in file_data["contenu"].lower():
                    files.append(FileData(**file_data))

    if sort_by_date:
        files.sort(key=lambda x: x.date_creation, reverse=True)
    else:
        files.sort(key=lambda x: x.id)

    return files[skip : skip + limit]

@app.post("/match_cv_job/", response_model=MatchResult)
async def match_cv_to_job_offer(match_request: MatchRequest):
    cv_file = os.path.join(CANDIDATURES_DIR, f"{match_request.cv_id}.json")
    job_file = os.path.join(OFFRES_DIR, f"{match_request.job_id}.json")

    if not os.path.exists(cv_file) or not os.path.exists(job_file):
        raise HTTPException(status_code=404, detail="CV ou offre d'emploi non trouvé")

    try:
        with open(cv_file, "r") as f:
            cv_data = FileData(**json.load(f))
        with open(job_file, "r") as f:
            job_data = FileData(**json.load(f))

        print(f"Longueur du contenu CV: {len(cv_data.contenu)}")
        print(f"Longueur du contenu offre: {len(job_data.contenu)}")

        cv_analysis, job_analysis = await asyncio.gather(
            extract_skills_and_info(cv_data.contenu, is_cv=True),
            extract_skills_and_info(job_data.contenu, is_cv=False)
        )

        print(f"Analyse CV: {cv_analysis[:100]}...")
        print(f"Analyse offre: {job_analysis[:100]}...")

        match_result = await compare_cv_to_job(cv_analysis, job_analysis)

        cv_embedding = get_embedding(cv_analysis)
        job_embedding = get_embedding(job_analysis)

        similarity_score = cosine_similarity([cv_embedding], [job_embedding])[0][0]
        combined_score = (float(match_result['score']) + similarity_score * 100) / 2

        result = MatchResult(
            cv_id=match_request.cv_id,
            job_id=match_request.job_id,
            score=combined_score,
            correspondances=match_result['correspondances'],
            non_correspondances=match_result['non_correspondances'],
            justification=match_result['justification'],
            cv_summary=cv_analysis,
            job_summary=job_analysis
        )

        return result
    except Exception as e:
        print(f"Erreur détaillée lors du matching: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Erreur lors du matching: {str(e)}")

@app.on_event("startup")
async def startup_event():
    FastAPICache.init(InMemoryBackend())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=4)
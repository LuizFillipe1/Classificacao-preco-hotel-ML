from app.views import router
from fastapi import FastAPI

# Definir o aplicativo FastAPI
app = FastAPI()

# Incluir as rotas
app.include_router(router)

# Para rodar a aplicação com Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

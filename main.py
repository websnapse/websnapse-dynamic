from fastapi import FastAPI
from models import SNPSystem
from fastapi.middleware.cors import CORSMiddleware
from SNP import MatrixSNPSystem

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:5173",
]
app.add_middleware(  # type: ignore
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/simulate")
async def simulate_all(system: SNPSystem):
    matrixSNP = MatrixSNPSystem(system)
    matrixSNP.simulate_all()

    print(matrixSNP.states)
    print(matrixSNP.contents)

    return {
        "states": matrixSNP.states.tolist(),
        "configurations": matrixSNP.contents.tolist(),
        "keys": matrixSNP.neuron_keys,
    }


@app.post("/simulate/last")
async def simulate_all(system: SNPSystem):
    matrixSNP = MatrixSNPSystem(system)
    matrixSNP.simulate_all()

    print(matrixSNP.content)

    return {
        "contents": matrixSNP.content.tolist(),
    }


@app.post("/simulate/step")
async def simulate_step(system: SNPSystem):
    matrixSNP = MatrixSNPSystem(system)
    matrixSNP.simulate()

    print(matrixSNP.state)
    print(matrixSNP.content)
    return {
        "states": matrixSNP.state.tolist(),
        "configurations": matrixSNP.content.tolist(),
        "keys": matrixSNP.neuron_keys,
    }

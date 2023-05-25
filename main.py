import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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
    matrixSNP.simulate_pseudorandom()

    print(matrixSNP.state)
    print(matrixSNP.content)
    print(matrixSNP.halted)
    return {
        "states": matrixSNP.state.tolist(),
        "configurations": matrixSNP.content.tolist(),
        "keys": matrixSNP.neuron_keys,
        "halted": bool(matrixSNP.halted),
    }


@app.post("/check")
async def check(system: SNPSystem):
    matrixSNP = MatrixSNPSystem(system)

    matrixSNP.compute_spikeable_mx()
    print(matrixSNP.spikeable_mx)

    spikeable = matrixSNP.check_non_determinism()
    print(spikeable)
    choice = {
        "n1": 0,
        "n2": 1,
        "n3": 0,
    }
    print(choice)
    spiking_vector = matrixSNP.create_spiking_vector(choice)
    print(spiking_vector)


@app.websocket("/simulate/ws")
async def websocket_endpoint(websocket: WebSocket):
    matrixSNP = None
    status = {
        "1": "animate",
        "0": "default",
        "-1": "closed",
    }

    await websocket.accept()

    data = await websocket.receive_json()
    matrixSNP = MatrixSNPSystem(SNPSystem(**data))
    while True:
        try:
            await asyncio.sleep(1)

            matrixSNP.simulate_pseudorandom()

            configs = {}
            states = {}
            for key, state, content in zip(
                matrixSNP.neuron_keys, matrixSNP.state, matrixSNP.content
            ):
                configs[key] = content
                states[key] = status[str(state)]

            await websocket.send_json(
                {
                    "states": states,
                    "configurations": configs,
                    "halted": bool(matrixSNP.halted),
                }
            )
            if matrixSNP.halted:
                break
        except:
            websocket.close()
            break

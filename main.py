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
    matrixSNP.pseudorandom_simulate_all()

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
    matrixSNP.pseudorandom_simulate_all()

    print(matrixSNP.content)

    return {
        "contents": matrixSNP.content.tolist(),
    }


@app.post("/simulate/step")
async def simulate_step(system: SNPSystem):
    matrixSNP = MatrixSNPSystem(system)
    matrixSNP.compute_next_configuration()

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
    matrixSNP.create_spiking_vector(choice)
    matrixSNP.compute_next_configuration()
    print(matrixSNP.state)
    print(matrixSNP.content)
    print(matrixSNP.halted)


@app.websocket("/ws/simulate/guided")
async def guided_mode(websocket: WebSocket):
    status = {
        "1": "spiking",
        "0": "default",
        "-1": "closed",
    }

    await websocket.accept()

    data = await websocket.receive_json()
    matrixSNP = MatrixSNPSystem(SNPSystem(**data))
    while True:
        try:
            await asyncio.sleep(1)
            matrixSNP.compute_spikeable_mx()
            choices = matrixSNP.spikeable_mx.shape[0]
            if choices > 1:
                spikeable = matrixSNP.check_non_determinism()
                await websocket.send_json({"type": "prompt", "choices": spikeable})
                choice = await websocket.receive_json()
                matrixSNP.create_spiking_vector(choice)
            else:
                matrixSNP.decision_vct = matrixSNP.spikeable_mx[0]
            matrixSNP.compute_next_configuration()

            configs = {}
            states = {}
            for key, state, content in zip(
                matrixSNP.neuron_keys, matrixSNP.state, matrixSNP.content
            ):
                configs[key] = content
                states[key] = status[str(state)]

            await websocket.send_json(
                {
                    "type": "step",
                    "states": states,
                    "configurations": configs,
                    "halted": bool(matrixSNP.halted),
                }
            )
            if matrixSNP.halted:
                print("stop")
                break
        except:
            websocket.close()
            break


@app.websocket("/ws/simulate/pseudorandom")
async def pseudorandom_mode(websocket: WebSocket):
    status = {
        "1": "spiking",
        "0": "default",
        "-1": "closed",
    }

    await websocket.accept()

    data = await websocket.receive_json()
    matrixSNP = MatrixSNPSystem(SNPSystem(**data))
    while True:
        try:
            await asyncio.sleep(1)
            matrixSNP.pseudorandom_simulate_next()

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

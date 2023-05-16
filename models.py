from pydantic import BaseModel, Field
from typing import Literal, List, Union


class InputNeuron(BaseModel):
    id: str
    nodeType: Literal["input"]
    spiketrain: str


class OutputNeuron(BaseModel):
    id: str
    nodeType: Literal["output"]


class RegularNeuron(BaseModel):
    id: str
    nodeType: Literal["regular"]
    content: int
    rules: List[str]


class Synapse(BaseModel):
    source: str
    target: str
    label: float


class SNPSystem(BaseModel):
    nodes: List[Union[InputNeuron, OutputNeuron, RegularNeuron]]
    edges: List[Synapse]

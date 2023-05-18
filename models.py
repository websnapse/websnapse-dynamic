from pydantic import BaseModel
from typing import Literal, List, Union, Optional


class InputNeuron(BaseModel):
    id: str
    type: Literal["input"]
    spiketrain: str


class OutputNeuron(BaseModel):
    id: str
    type: Literal["output"]
    spiketrain: str


class RegularNeuron(BaseModel):
    id: str
    type: Literal["regular"]
    content: int
    rules: List[str]


class Synapse(BaseModel):
    source: str
    target: str
    label: float


class SNPSystem(BaseModel):
    nodes: List[Union[InputNeuron, OutputNeuron, RegularNeuron]]
    edges: List[Synapse]
    expected: Optional[List[object]]

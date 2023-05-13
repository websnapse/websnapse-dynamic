from pydantic import BaseModel
from typing import Literal, List, Optional


class Neuron(BaseModel):
    id: str
    content: int
    nodeType: Literal["input", "output", "regular"]
    rules: Optional[List[str]] = None


class Synapse(BaseModel):
    source: str
    target: str
    label: float


class SNPSystem(BaseModel):
    nodes: List[Neuron]
    edges: List[Synapse]

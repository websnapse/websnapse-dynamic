from pydantic import BaseModel, Field
from typing import Literal, List, Union, Optional


class Node(BaseModel):
    id: str
    type: Literal["input", "output", "regular"]
    content: Union[int, str]
    rules: Optional[List[str]]


class Synapse(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    weight: float


class SNPSystem(BaseModel):
    nodes: List[Node]
    synapses: List[Synapse]
    expected: Optional[List[object]]

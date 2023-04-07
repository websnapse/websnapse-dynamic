from pydantic import BaseModel
import regex as re


class Rule(BaseModel):
    id: str
    definition: str


def parseRule(self, rule):
    left, right = rule.definition.split('->')
    bound = None
    if '/' in left:
        bound, left = left.split('/')

    consumption = -1 if '^' not in left else -int(left.split('^')[1])
    production = 0 if right == 'λ' else 1 if '^' not in right else int(
        right.split('^')[1])


rule = Rule(id='1', definition="a/a->λ")
rule.definition = rule.definition.replace('λ', 'a^{0};1')
pattern = r"^(.*)\/a(\^{(\d+)})?->a(\^{(\d+)})?;(\d*)"
result = re.match(pattern, rule.definition)

if result:

    bound = result.group(1) or ''
    consumption = result.group(3) or -1
    production = result.group(5) or 1
    delay = int(result.group(6))

    consumption = int(consumption)
    production = int(production)

    print(bound, consumption, production, delay)

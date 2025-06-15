from parsimonious import Grammar, NodeVisitor
from collections import namedtuple
from dataclasses import dataclass        
                

grammar_text = (r'''
formula = ( _ paren_formula _ ) / ( _ pixel_formula _ ) / ( _ region_formula _ ) 
paren_formula = "(" _ formula _ ")"
pixel_formula = ( _ paren_pixel_formula _ ) / ( _ top _ ) / ( _ pixel_atomic _ ) / ( _ neg _ ) / ( _ and _ ) / ( _ or _ ) / ( _ implies _ ) / ( _ next _ )
paren_pixel_formula = "(" _ pixel_formula _ ")"
region_formula = ( _ paren_region_formula _ ) / ( _ exists _ ) / ( _ forall _ ) / ( _ neg _ ) / ( _ and _ ) / ( _ or _ ) / ( _ implies _ ) / ( _ solidtriangle _ ) / ( _ hollowtriangle _ )
paren_region_formula = "(" _ region_formula _ ")"
pixel_atomic = ( _ paren_pixel_atomic _) / ( _ id_atomic _ ) / ( _ class_atomic _ ) / ( _ prob_atomic _ ) / ( _ intensity_atomic _ ) / ( _ row_atomic _ ) / ( _ col_atomic _ ) 
paren_pixel_atomic = "(" _ pixel_atomic _ ")"
id_atomic = "p" _ optional_pixel_identifier _ "." _ "id" _ "==" _ int _
class_atomic = "p" _ optional_pixel_identifier _ "." _ "class" _ "==" _ int _
prob_atomic = "p" _ optional_pixel_identifier _ "." _ "prob" _ comparator _ float _
intensity_atomic = "p" _ optional_pixel_identifier _ "." _ "I" _ comparator _ float _
row_atomic = "p" _ optional_pixel_identifier _ "." _ "row" _ comparator _ float _
col_atomic = "p" _ optional_pixel_identifier _ "." _ "col" _ comparator _ float _
next = "O" _ direction _ pixel_formula 
exists = "exists" _ region_identifier _ pixel_formula   # beginning of region formulas
forall = "forall" _ region_identifier _ pixel_formula
solidtriangle = "solidtriangle" _ direction _ step _ region_identifier _ region_formula
hollowtriangle = "hollowtriangle" _ direction _ step _ region_identifier _ region_formula
neg = "!" _ formula
and = "(" _ formula _ "&" _ formula _ ")"
or = "(" _ formula _ "|" _ formula _ ")"
implies = "(" _ formula _ "->" _ formula _ ")"
top = "True"
optional_pixel_identifier = pixel_identifier / ""
pixel_identifier = "(" _ int _ "," _ int _ ")" 
region_identifier = "(" _ pixel_identifier _ "," _ pixel_identifier _ "," _ pixel_identifier _ "," _ pixel_identifier _ ")"
direction = ~"[NSEW]"
step = "[" _ int _ "]"
comparator = "<=" / ">=" / "<" / ">" / "="
identifier = ~r"[a-zA-Z_][a-zA-Z0-9_]*"
int = ~r"[0-9]+"
float = ~r"[0-9]+(\.[0-9]+)?"
_ = ~r"\s*"
''')



_grammar = Grammar(grammar_text)



class DSCLLogicVisitor(NodeVisitor):

    def visit_formula(self, node, children):
        return children[0][1]

    def visit_paren_formula(self, node, children):
        return children[2]
    
    def visit_pixel_formula(self, node, children):
        return children[0][1]
    
    def visit_paren_pixel_formula(self, node, children):
        return children[2]
        
    def visit_region_formula(self, node, children):
        return children[0][1]
    
    def visit_paren_region_formula(self, node, children):
        return children[2]
        
    def visit_pixel_atomic(self, node, children):
        return children[0][1]
    
    def visit_paren_pixel_atomic(self, node, children):
        return children[2]
        
    def visit_id_atomic(self, node, children):
        _,_,pixel_identifier,_,_,_,_,_,_,_,value,_ = children
        return IdAtomic(pixel_identifier, value)
        
    def visit_class_atomic(self, node, children):
        _,_,pixel_identifier,_,_,_,_,_,_,_,value,_ = children
        return ClassAtomic(pixel_identifier, value)
        
    def visit_prob_atomic(self, node, children):
        _,_,pixel_identifier,_,_,_,_,_,comparator,_,value,_ = children
        return ProbAtomic(pixel_identifier, comparator, value)
        
    def visit_intensity_atomic(self, node, children):
        _,_,pixel_identifier,_,_,_,_,_,comparator,_,value,_ = children
        return IntensityAtomic(pixel_identifier, comparator, value)
        
    def visit_row_atomic(self, node, children):
        _,_,pixel_identifier,_,_,_,_,_,comparator,_,value,_ = children
        return RowAtomic(pixel_identifier, comparator, value)
        
    def visit_col_atomic(self, node, children):
        _,_,pixel_identifier,_,_,_,_,_,comparator,_,value,_ = children
        return ColAtomic(pixel_identifier, comparator, value)
        
    def visit_next(self, node, children):
        _,_,direction,_,subformula = children
        return Next(direction, subformula)
        
    def visit_exists(self, node, children):
        _,_,region_identifier,_,subformula = children 
        return Exists(region_identifier, subformula)
        
    def visit_forall(self, node, children):
        _,_,region_identifier,_,subformula = children 
        return Forall(region_identifier, subformula)
        
    def visit_solidtriangle(self, node, children):
        _,_,direction,_,step,_,region_identifier,_,subformula = children
        return StrongDistance(direction, step, region_identifier, subformula)
        
    def visit_hollowtriangle(self, node, children):
        _,_,direction,_,step,_,region_identifier,_,subformula = children
        return WeakDistance(direction, step, region_identifier, subformula)
        
    def visit_neg(self, node, children):
        _,_,subformula = children
        return Negation(subformula)
        
    def visit_and(self, node, children):
        _, _, left, _, _, _, right, _, _ = children
        return And(left, right)
        
    def visit_or(self, node, children):
        _, _, left, _, _, _, right, _, _ = children
        return Or(left, right)
        
    def visit_implies(self, node, children):
        _, _, left, _, _, _, right, _, _ = children
        return Implies(left, right)
        
    def visit_top(self, node, children):
        ...
        
    def visit_optional_pixel_identifier(self, node, children):
        if not children[0]:
            return PixelIdentifier(-1000, -1000)
        else:
            return children[0]
        
    def visit_pixel_identifier(self, node, children):
        _,_,row,_,_,_,col,_,_ = children 
        return PixelIdentifier(row, col)
    
    def visit_region_identifier(self, node, children):
        _,_,p1,_,_,_,p2,_,_,_,p3,_,_,_,p4,_,_ = children
        return RegionIdentifier(p1,p2,p3,p4)
        
    def visit_direction(self, node, children):
        return ConstantStr(node.text)
    
    def visit_step(self, node, children):
        _, _, step, _, _ = children
        return Step(step)
        
    def visit_comparator(self, node, children):
        return ConstantStr(node.text)
        
    def visit_identifier(self, node, children):
        return ConstantStr(node.text)
        
    def visit_int(self, node, children):
        return ConstantInt(node.text)
        
    def visit_float(self, node, children):
        return Constant(node.text)
    
    def generic_visit(self, node, children):
        if children:
            return children


@dataclass
class IdAtomic:
    pixel_identifier: object
    value: int

    def __repr__(self):
        if (self.pixel_identifier.row, self.pixel_identifier.col) == (-1000, -1000):
            return f"p.id=={self.value}"
        else:
            return f"p({self.pixel_identifier.row},{self.pixel_identifier.col}).id=={self.value}"


@dataclass
class ClassAtomic:
    pixel_identifier: object
    value: int

    def __repr__(self):
        if (self.pixel_identifier.row, self.pixel_identifier.col) == (-1000, -1000):
            return f"p.class=={self.value}"
        else:
            return f"p({self.pixel_identifier.row},{self.pixel_identifier.col}).class=={self.value}"


@dataclass
class ProbAtomic:
    pixel_identifier: object
    comparator: str
    value: float

    def __repr__(self):
        if (self.pixel_identifier.row, self.pixel_identifier.col) == (-1000, -1000):
            return f"p.prob{self.comparator}{self.value}"
        return f"p({self.pixel_identifier.row},{self.pixel_identifier.col}).prob{self.comparator}{self.value}"


@dataclass
class IntensityAtomic:
    pixel_identifier: object
    comparator: str
    value: float

    def __repr__(self):
        if (self.pixel_identifier.row, self.pixel_identifier.col) == (-1000, -1000):
            return f"p.I{self.comparator}{self.value}"
        return f"p({self.pixel_identifier.row},{self.pixel_identifier.col}).I{self.comparator}{self.value}"


@dataclass
class RowAtomic:
    pixel_identifier: object
    comparator: str
    value: int

    def __repr__(self):
        if (self.pixel_identifier.row, self.pixel_identifier.col) == (-1000, -1000):
            return f"p.row{self.comparator}{self.value}"
        return f"p({self.pixel_identifier.row},{self.pixel_identifier.col}).row{self.comparator}{self.value}"


@dataclass
class ColAtomic:
    pixel_identifier: object
    comparator: str
    value: int

    def __repr__(self):
        if (self.pixel_identifier.row, self.pixel_identifier.col) == (-1000, -1000):
            return f"p.col{self.comparator}{self.value}"
        return f"p({self.pixel_identifier.row},{self.pixel_identifier.col}).col{self.comparator}{self.value}"


@dataclass
class Next:
    direction: str
    subformula: object

    def __repr__(self):
        return f"O({self.direction}){self.subformula}"

    def children(self):
        return [self.subformula]
    
    
@dataclass
class Exists:
    region_identifier: object 
    subformula: object 
    
    def __repr__(self):
        return f"exists{self.region_identifier}({self.subformula})"

    def children(self):
        return [self.subformula]
    
    
@dataclass
class Forall: 
    region_identifier: object 
    subformula: object 
    
    def __repr__(self):
        return f"forall{self.region_identifier}({self.subformula})"

    def children(self):
        return [self.subformula]
    
    
@dataclass
class StrongDistance:
    direction: str
    step: int
    region_identifier: object
    subformula: object
    
    def __repr__(self):
        return f"solidtriangle({self.direction},{self.step}) {self.region_identifier} {self.subformula}"
    
    def children(self): 
        return [self.left, self.right]
    
    
@dataclass
class WeakDistance:
    direction: str
    step: int
    region_identifier: object
    subformula: object
    
    def __repr__(self):
        return f"hollowtriangle({self.direction},{self.step}) {self.region_identifier} {self.subformula}"
    
    def children(self): 
        return [self.left, self.right]
    
    
@dataclass
class Negation:
    subformula: object

    def __repr__(self):
        return f"(! {self.subformula})"

    def children(self):
        return [self.subformula]


@dataclass
class And:
    left: object
    right: object

    def __repr__(self):
        return f"({self.left} & {self.right})"

    def children(self):
        return [self.left, self.right]


@dataclass
class Or:
    left: object
    right: object

    def __repr__(self):
        return f"({self.left} | {self.right})"

    def children(self):
        return [self.left, self.right]


@dataclass
class Implies:
    left: object
    right: object

    def __repr__(self):
        return f"({self.left} -> {self.right})"

    def children(self):
        return [self.left, self.right]
    
    
@dataclass
class PixelIdentifier:
    row: int
    col: int
    
    def __repr__(self):
        return f"({self.row}, {self.col})"
    
    
@dataclass
class RegionIdentifier:
    p1: object
    p2: object
    p3: object
    p4: object
    
    def __repr__(self):
        return f"({self.p1}, {self.p2}, {self.p3}, {self.p4})"
    
    
@dataclass
class Step:
    step: int
    
    def __repr__(self):
        return f"[{self.step}]"
    

class Constant(float):
    pass

class ConstantInt(int):
    pass

class ConstantStr(str):
    pass


def parse(input_str):
    return DSCLLogicVisitor().visit(_grammar.parse(input_str))




if __name__ == '__main__':
    # Example usage
    input_str = "p(1,2).id  ==  1"
    print("Input:\t", IdAtomic(PixelIdentifier(1,2), 1))
    print("Parsed:\t", DSCLLogicVisitor().visit(_grammar.parse(input_str)))
    
    input_str = "p.id  ==  1"
    print("Input:\t", IdAtomic(PixelIdentifier(-1000,-1000), 1))
    print("Parsed:\t", DSCLLogicVisitor().visit(_grammar.parse(input_str)))
    
    input_str = "  p(1,2).class  == 1"
    print("Input:\t", ClassAtomic(PixelIdentifier(1,2), "1"))
    print("Parsed:\t", DSCLLogicVisitor().visit(_grammar.parse(input_str)))
    
    input_str = "  p(1,2).prob <=  0.5"
    print("Input:\t", ProbAtomic(PixelIdentifier(1,2), "<", 0.5))
    print("Parsed:\t", DSCLLogicVisitor().visit(_grammar.parse(input_str)))
    
    input_str = "  p(1,2).prob >= 0.5"
    print("Input:\t", ProbAtomic(PixelIdentifier(1,2), ">=", 0.5))
    print("Parsed:\t", DSCLLogicVisitor().visit(_grammar.parse(input_str)))
    
    input_str = "  p(1,2).I >= 1000"
    print("Input:\t", IntensityAtomic(PixelIdentifier(1,2), ">=", 1000))
    print("Parsed:\t", DSCLLogicVisitor().visit(_grammar.parse(input_str)))
    
    input_str = "  p(1,2).row <= 2"
    print("Input:\t", RowAtomic(PixelIdentifier(1,2), "<=", 2))
    print("Parsed:\t", DSCLLogicVisitor().visit(_grammar.parse(input_str)))
    
    input_str = " O N p(1,2).prob >= 0.5"
    input_str_1 = " p(1,2).prob >= 0.5 "
    print("Input:\t", Next("N", input_str_1))
    print("Parsed:\t", DSCLLogicVisitor().visit(_grammar.parse(input_str)))
    
    input_str = "p(1,2).row <= 2"
    neg_input_str = "! p(1,2).row <= 2"
    print("Input:\t", Negation(input_str))
    print("Parsed:\t", DSCLLogicVisitor().visit(_grammar.parse(neg_input_str)))
    
    infer_str = "({} -> {})".format(input_str, input_str_1)
    print("Parsed:\t", DSCLLogicVisitor().visit(_grammar.parse(infer_str)))
    
    or_input_str = "(p(1,2).row <= 2 | ! p(1,2).row <= 2 )"
    print("Input:\t", Or(input_str, neg_input_str))
    print("Parsed:\t", DSCLLogicVisitor().visit(_grammar.parse(or_input_str)))
    
    input_str = "exists ((0, 0), (0, 1), (1, 1), (1, 0)) p.id == 2"
    p1 = PixelIdentifier(0,0)
    p2 = PixelIdentifier(0,1)
    p3 = PixelIdentifier(1,1)
    p4 = PixelIdentifier(1,0)
    px_formula = "p.id == 2"
    region_id = RegionIdentifier(p1, p2, p3, p4)
    print("Input:\t", Exists(region_id, px_formula))
    print("Parsed:\t", DSCLLogicVisitor().visit(_grammar.parse(input_str)))
    
    input_str = "forall ((0, 0), (0, 1), (1, 1), (1, 0)) p.row >= 2"
    px_formula = "p.row >= 2"
    print("Input:\t", Forall(region_id, px_formula))
    print("Parsed:\t", DSCLLogicVisitor().visit(_grammar.parse(input_str)))
    
    input_str = "solidtriangle N [10] ((0, 0), (0, 1), (1, 1), (1, 0)) (exists ((0, 0), (0, 1), (1, 1), (1, 0)) p.class == 1)"
    right_formula = "p.class == 1"
    print("Input:\t", StrongDistance("N", 10, region_id , Exists(region_id, right_formula)))
    print("Parsed:\t", DSCLLogicVisitor().visit(_grammar.parse(input_str)))
    
    
    print("Done.")
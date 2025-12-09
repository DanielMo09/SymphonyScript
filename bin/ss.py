from dataclasses import dataclass
from typing import List, Union, Optional
import re
import sys
import argparse
import logging
import os
import datetime

argumentparser = argparse.ArgumentParser("ss.py")
argumentparser.add_argument("-d", "--debug", help="Debug mode, shows more info", type=bool)
argumentparser.add_argument("-i", "--input", help="The code input file", type=str)
argumentparser.add_argument("-o", "--output", help="The ASM output file (defaultly outputs to console)", type=str)
args = argumentparser.parse_args()

if not args.input or not args.output:
    argumentparser.print_help()
    sys.exit(0)

logger = logging.Logger("SympthonyScript")
logging.basicConfig(handlers=(logging.FileHandler(os.path.join(os.path.dirname(__file__), "..", "log", "log " + str(datetime.datetime.now().strftime("%d.%m.%Y %H.%M.%S")) + ".log"), "w"), logging.StreamHandler()))
try:
    if args.debug != False:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
except:
    logger.setLevel(logging.INFO)

# -------------------------
# AST Node Definitions
# -------------------------
@dataclass
class Node:
    pass

@dataclass
class Program(Node):
    statements: List['Statement']

@dataclass
class Statement(Node):
    pass

@dataclass
class VarDecl(Statement):
    name: str
    value: Optional['Expr']

@dataclass
class Assignment(Statement):
    name: str
    value: 'Expr'

@dataclass
class RawASM(Statement):
    code: str

@dataclass
class IfStmt(Statement):
    condition: 'BinaryOp'
    body: List['Statement']

@dataclass
class FunctionDecl(Statement):
    name: str
    body: List['Statement']

@dataclass
class ReturnStmt(Statement):
    pass

@dataclass
class Expr(Node):
    pass

@dataclass
class BinaryOp(Expr):
    op: str
    left: Expr
    right: Expr

@dataclass
class Number(Expr):
    value: int

@dataclass
class VarRef(Expr):
    name: str

# -------------------------
# Lexer + Parser
# -------------------------
TOKEN_RE = re.compile(r'\s*(\d+|[A-Za-z_][A-Za-z0-9_]*|<<|>>|==|!=|<=|>=|=|[+\-*/{}();])')

class CLexer:
    def __init__(self, code: str):
        self.tokens = []
        self.pos = 0
        self.tokenize(code)

    def tokenize(self, code: str):
        for tok in TOKEN_RE.findall(code):
            if tok.isdigit():
                self.tokens.append(('NUMBER', tok))
            elif re.match(r'[A-Za-z_][A-Za-z0-9_]*', tok):
                self.tokens.append(('IDENT', tok))
            else:
                self.tokens.append(('SYM', tok))
        self.tokens.append(('EOF',''))
        logger.debug('DEBUG TOKENS:', self.tokens)

    def peek(self):
        return self.tokens[self.pos]

    def next(self):
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

class CParser:
    def __init__(self, lexer: CLexer):
        self.lexer = lexer
        self.if_counter = 0

    def parse_program(self) -> Program:
        stmts = []
        while self.lexer.peek()[0] != 'EOF':
            stmt = self.parse_statement()
            if stmt:
                stmts.append(stmt)
        logger.debug('DEBUG AST:', stmts)
        return Program(statements=stmts)

    def parse_statement(self) -> Optional[Statement]:
        tok_type, tok_val = self.lexer.peek()
        if tok_type == 'IDENT':
            ident = tok_val
            self.lexer.next()

            if ident == 'int' and self.lexer.peek()[0] == 'IDENT':
                var_name = self.lexer.next()[1]
                value = None
                if self.lexer.peek()[1] == '=':
                    self.lexer.next()
                    value = self.parse_expression()
                if self.lexer.peek()[1] == ';':
                    self.lexer.next()
                logger.debug('DEBUG VARDECL:', var_name, value)
                return VarDecl(name=var_name, value=value)

            elif ident == 'if':
                left = self.parse_expression()
                op_tok = self.lexer.next()
                if op_tok[1] in ('==', '=', '!=', '<', '>', '<=', '>='):
                    right = self.parse_expression()
                    condition = BinaryOp(op=op_tok[1], left=left, right=right)
                else:
                    condition = left
                body = []
                if self.lexer.peek()[1] == '{':
                    self.lexer.next()
                    while self.lexer.peek()[1] != '}' and self.lexer.peek()[0] != 'EOF':
                        stmt = self.parse_statement()
                        if stmt:
                            body.append(stmt)
                    if self.lexer.peek()[1] == '}':
                        self.lexer.next()
                self.if_counter += 1
                logger.debug('DEBUG IF:', condition, 'body:', body)
                return IfStmt(condition=condition, body=body)

            elif ident == 'asm' and self.lexer.peek()[1] == '(':
                self.lexer.next()
                code_tokens = []
                while self.lexer.peek()[1] != ')':
                    code_tokens.append(self.lexer.next()[1])
                self.lexer.next()
                if self.lexer.peek()[1] == ';':
                    self.lexer.next()
                code_str = ' '.join(code_tokens)
                logger.debug('DEBUG ASM:', code_str)
                return RawASM(code=code_str)

        self.lexer.next()
        return None

    def parse_expression(self) -> Expr:
        tok_type, tok_val = self.lexer.next()
        if tok_type == 'NUMBER':
            return Number(int(tok_val))
        elif tok_type == 'IDENT':
            expr = VarRef(tok_val)
            if self.lexer.peek()[1] in ('+','-','*','<<','>>'):
                op = self.lexer.next()[1]
                right = self.parse_expression()
                return BinaryOp(op=op, left=expr, right=right)
            return expr
        return VarRef(tok_val)

# -------------------------
# Code Generator
# -------------------------
REGISTER_MAP = ['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','r11','r12','r13']
var_to_reg = {}
var_to_mem = {}
reg_index = 4
mem_addr_counter = 0x1000
RAM_THRESHOLD = 10
_if_label_counter = 0
_func_label_counter = 0

def use_ram():
    total_vars = len(var_to_reg) + len(var_to_mem)
    return total_vars >= RAM_THRESHOLD

def gen_expr(expr: Expr, target_reg: str) -> List[str]:
    asm = []
    if isinstance(expr, Number):
        asm.append(f'mov {target_reg}, {expr.value}')
    elif isinstance(expr, VarRef):
        if expr.name in var_to_reg:
            asm.append(f'mov {target_reg}, {var_to_reg[expr.name]}')
        elif expr.name in var_to_mem:
            asm.append(f'load_16 {target_reg}, [{var_to_mem[expr.name]}]')
        else:
            asm.append(f'; WARNING: variable {expr.name} not found')
    elif isinstance(expr, BinaryOp):
        left_reg, right_reg = 'r1','r2'
        asm += gen_expr(expr.left, left_reg)
        asm += gen_expr(expr.right, right_reg)
        op_map = {'+':'add','-':'sub','*':'mul','<<':'lsl','>>':'lsr','=':'cmp'}
        if expr.op in ('=', '=='):
            asm.append(f'cmp {left_reg}, {right_reg}')
        else:
            asm.append(f'{op_map.get(expr.op,"add")} r3, {left_reg}, {right_reg}')
    return asm

def gen_statement(stmt: Statement) -> List[str]:
    global reg_index, mem_addr_counter, _if_label_counter, _func_label_counter
    asm: List[str] = []

    if isinstance(stmt, RawASM):
        asm.append(stmt.code)

    elif isinstance(stmt, VarDecl):
        if use_ram():
            var_to_mem[stmt.name] = mem_addr_counter
            mem_addr_counter += 2
            if stmt.value:
                asm += gen_expr(stmt.value, 'r3')
                asm.append(f'store_16 [{var_to_mem[stmt.name]}], r3')
        else:
            reg = REGISTER_MAP[reg_index]
            var_to_reg[stmt.name] = reg
            reg_index += 1
            if stmt.value:
                asm += gen_expr(stmt.value, reg)

    elif isinstance(stmt, Assignment):
        if stmt.name in var_to_reg:
            asm += gen_expr(stmt.value, var_to_reg[stmt.name])
        elif stmt.name in var_to_mem:
            asm += gen_expr(stmt.value, 'r3')
            asm.append(f'store_16 [{var_to_mem[stmt.name]}], r3')
        else:
            reg = REGISTER_MAP[reg_index]
            var_to_reg[stmt.name] = reg
            reg_index += 1
            asm += gen_expr(stmt.value, reg)

    elif isinstance(stmt, IfStmt):
        label = f'if_skip{_if_label_counter}'
        _if_label_counter += 1
        if isinstance(stmt.condition, BinaryOp) and stmt.condition.op in ('==', '!=', '<', '>', '<=', '>='):
            asm += gen_expr(stmt.condition.left, 'r1')
            asm += gen_expr(stmt.condition.right, 'r2')
            op_map = {'==':'jne','=':'jne','!=':'je','<':'jge','<=':'jg','>':'jle','>=':'jl'}
            asm.append(f'{op_map[stmt.condition.op]} {label}')
            for s in stmt.body:
                asm += gen_statement(s)
            asm.append(f'{label}:')

    elif isinstance(stmt, FunctionDecl):
        asm.append(f'{stmt.name}:')
        for s in stmt.body:
            asm += gen_statement(s)
        asm.append(f'{stmt.name}_return:')
        asm.append('ret')

    elif isinstance(stmt, ReturnStmt):
        asm.append('ret')

    return asm

def gen_program(program: Program) -> List[str]:
    asm: List[str] = []
    for stmt in program.statements:
        asm += gen_statement(stmt)
    return asm

if __name__ == '__main__':
    
    
    with open(args.input, "r") as f:
        code = f.read()
    lexer = CLexer(code)
    parser = CParser(lexer)
    program = parser.parse_program()
    logger.debug('DEBUG AST:', program)

    asm_output = gen_program(program)
    if not args.output:
        for line in asm_output:
            print(line)
        sys.exit(0)

    with open(args.output, "w") as f:
        for line in asm_output:
            f.write(line + "\n")
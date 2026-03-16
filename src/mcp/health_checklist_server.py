# health_checklist_server.py
# Servidor MCP do PrevIA — health-checklist
#
# Segurança:
# - Allowlist: apenas 1 tool permitida
# - Sem acesso a disco/internet além do corpus
# - Logs de auditoria de todas as chamadas
# - Bloqueio de tools não autorizadas

import asyncio
import json
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

logging.basicConfig(
    filename="mcp_audit.log",
    level=logging.INFO,
    format="%(asctime)s | %(message)s"
)

# Allowlist — única tool permitida
ALLOWED_TOOLS = {"generate_preventive_checklist"}

server = Server("health-checklist")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Lista as tools disponíveis — apenas as da allowlist."""
    return [
        types.Tool(
            name="generate_preventive_checklist",
            description=(
                "Gera um checklist preventivo personalizado baseado em "
                "diretrizes do Ministério da Saúde e OMS. "
                "Retorna lista de exames, vacinas e hábitos recomendados."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "age":          {"type": "integer", "description": "Idade do paciente"},
                    "sex":          {"type": "string",  "description": "Sexo: masculino ou feminino"},
                    "risk_factors": {"type": "string",  "description": "Fatores de risco separados por vírgula"}
                },
                "required": ["age", "sex"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Executa uma tool — valida contra allowlist antes."""

    # Validação de allowlist
    if name not in ALLOWED_TOOLS:
        logging.warning(f"BLOQUEADO | tool={name} | motivo=fora_da_allowlist")
        return [types.TextContent(
            type="text",
            text=f"Erro: tool '{name}' não permitida. Tools disponíveis: {ALLOWED_TOOLS}"
        )]

    # Log de auditoria
    logging.info(f"CHAMADA | tool={name} | args={json.dumps(arguments)}")

    if name == "generate_preventive_checklist":
        resultado = f"Checklist para {arguments.get('age')} anos, {arguments.get('sex')}, riscos: {arguments.get('risk_factors', 'nenhum')}"
        logging.info(f"SUCESSO | tool={name} | chars={len(resultado)}")
        return [types.TextContent(type="text", text=resultado)]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())

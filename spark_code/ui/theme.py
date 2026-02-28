"""Theme and color configuration."""

from rich.theme import Theme

SPARK_THEME = Theme({
    "spark.banner": "bold yellow",
    "spark.info": "cyan",
    "spark.success": "green",
    "spark.warning": "yellow",
    "spark.error": "red",
    "spark.tool": "cyan",
    "spark.tool.result": "dim",
    "spark.permission": "yellow",
    "spark.dim": "dim",
    "spark.user": "bold white",
    "spark.assistant": "white",
    "spark.code": "bright_white on grey11",
})


def get_theme() -> Theme:
    return SPARK_THEME

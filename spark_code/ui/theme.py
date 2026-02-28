"""Theme and color configuration — matches Claude Code's palette."""

from rich.theme import Theme

# Claude Code's actual color palette
SPARK_THEME = Theme({
    # Banner / chrome
    "spark.banner": "bold #e0ac69",
    "spark.banner.dim": "#666666",

    # Content
    "spark.info": "#88c0d0",
    "spark.success": "#a3be8c",
    "spark.warning": "#ebcb8b",
    "spark.error": "#bf616a",

    # Tool display
    "spark.tool": "bold #88c0d0",
    "spark.tool.name": "bold #88c0d0",
    "spark.tool.args": "#d8dee9",
    "spark.tool.result": "#666666",
    "spark.tool.connector": "#4c566a",

    # Permissions
    "spark.permission": "#ebcb8b",

    # Text
    "spark.dim": "#666666",
    "spark.user": "bold #eceff4",
    "spark.assistant": "#d8dee9",
    "spark.code": "#eceff4 on #2e3440",

    # Status
    "spark.status": "#4c566a",
    "spark.status.key": "#5e81ac",
    "spark.status.value": "#888888",
    "spark.status.sep": "#3b4252",
})


def get_theme() -> Theme:
    return SPARK_THEME

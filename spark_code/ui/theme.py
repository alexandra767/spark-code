"""Theme and color configuration — matches Claude Code's palette."""

from rich.theme import Theme

# Nord dark theme (default)
DARK_THEME = Theme({
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
    "spark.tool.result": "#8899aa",
    "spark.tool.connector": "#7b88a1",

    # Permissions
    "spark.permission": "#ebcb8b",

    # Text
    "spark.dim": "#8899aa",
    "spark.user": "bold #eceff4",
    "spark.assistant": "#d8dee9",
    "spark.code": "#eceff4 on #2e3440",

    # Status
    "spark.status": "#7b88a1",
    "spark.status.key": "#5e81ac",
    "spark.status.value": "#a0aabb",
    "spark.status.sep": "#5a6577",
})

# Light theme
LIGHT_THEME = Theme({
    # Banner / chrome
    "spark.banner": "bold #c08030",
    "spark.banner.dim": "#888888",

    # Content
    "spark.info": "#0077aa",
    "spark.success": "#2e7d32",
    "spark.warning": "#c07000",
    "spark.error": "#c62828",

    # Tool display
    "spark.tool": "bold #0077aa",
    "spark.tool.name": "bold #0077aa",
    "spark.tool.args": "#333333",
    "spark.tool.result": "#555555",
    "spark.tool.connector": "#777777",

    # Permissions
    "spark.permission": "#c07000",

    # Text
    "spark.dim": "#777777",
    "spark.user": "bold #1a1a1a",
    "spark.assistant": "#333333",
    "spark.code": "#1a1a1a on #f0f0f0",

    # Status
    "spark.status": "#777777",
    "spark.status.key": "#0055aa",
    "spark.status.value": "#555555",
    "spark.status.sep": "#cccccc",
})

# Keep backward compatibility
SPARK_THEME = DARK_THEME


def get_theme(name: str = "dark") -> Theme:
    if name == "light":
        return LIGHT_THEME
    return DARK_THEME

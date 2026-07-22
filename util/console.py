BRIGHT_GREEN = "\033[92m"
RESET = "\033[0m"


def bright_green(text: str) -> str:
    return f"{BRIGHT_GREEN}{text}{RESET}"

from src.game import Screen, Game


def run() -> None:
    screen = Screen()
    game = Game(screen, "MAT501 - Coursework")
    while game.frame():
        pass

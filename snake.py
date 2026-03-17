import pygame
import random
import sys

# Initialize everything
pygame.init()

# Window setup
SCREEN_W = 640
SCREEN_H = 480
BLOCK = 20

screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("Snake Game")
clock = pygame.time.Clock()

# Fonts
font = pygame.font.Font(None, 36)
big_font = pygame.font.Font(None, 64)

# Colors
BG_COLOR = (0, 0, 0)
SNAKE_COLOR = (0, 255, 0)
SNAKE_HEAD_COLOR = (0, 200, 0)
FOOD_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)
GRAY_COLOR = (150, 150, 150)


def random_food(snake_body):
    """Place food randomly, not on the snake."""
    while True:
        x = random.randrange(0, SCREEN_W, BLOCK)
        y = random.randrange(0, SCREEN_H, BLOCK)
        if (x, y) not in snake_body:
            return (x, y)


def draw_text(text, f, color, center):
    surface = f.render(text, True, color)
    rect = surface.get_rect(center=center)
    screen.blit(surface, rect)


def game_loop():
    # Snake starts in the middle, not moving
    start_x = SCREEN_W // 2
    start_y = SCREEN_H // 2
    snake = [(start_x, start_y)]
    dx, dy = 0, 0
    started = False
    food = random_food(snake)
    score = 0
    alive = True
    speed = 10

    while True:
        # --- EVENT HANDLING ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if not alive:
                    if event.key == pygame.K_SPACE:
                        return "restart"
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                else:
                    if event.key in (pygame.K_UP, pygame.K_w):
                        if dy == 0:
                            dx, dy = 0, -BLOCK
                            started = True
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        if dy == 0:
                            dx, dy = 0, BLOCK
                            started = True
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        if dx == 0:
                            dx, dy = -BLOCK, 0
                            started = True
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        if dx == 0:
                            dx, dy = BLOCK, 0
                            started = True
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

        # --- UPDATE ---
        if alive and started:
            head_x, head_y = snake[0]
            new_head = (head_x + dx, head_y + dy)

            # Wall collision
            if (new_head[0] < 0 or new_head[0] >= SCREEN_W or
                    new_head[1] < 0 or new_head[1] >= SCREEN_H):
                alive = False

            # Self collision
            elif new_head in snake:
                alive = False

            else:
                snake.insert(0, new_head)
                if new_head == food:
                    score += 1
                    food = random_food(snake)
                    if score % 5 == 0:
                        speed += 1
                else:
                    snake.pop()

        # --- DRAW ---
        screen.fill(BG_COLOR)

        if not alive:
            draw_text("GAME OVER", big_font, FOOD_COLOR, (SCREEN_W // 2, SCREEN_H // 2 - 30))
            draw_text(f"Score: {score}", font, TEXT_COLOR, (SCREEN_W // 2, SCREEN_H // 2 + 20))
            draw_text("SPACE to play again | ESC to quit", font, GRAY_COLOR, (SCREEN_W // 2, SCREEN_H // 2 + 60))
        else:
            # Draw food
            pygame.draw.rect(screen, FOOD_COLOR, (food[0], food[1], BLOCK, BLOCK))

            # Draw snake
            for i, segment in enumerate(snake):
                color = SNAKE_HEAD_COLOR if i == 0 else SNAKE_COLOR
                pygame.draw.rect(screen, color, (segment[0], segment[1], BLOCK, BLOCK))

            # Draw score
            score_text = font.render(f"Score: {score}", True, TEXT_COLOR)
            screen.blit(score_text, (10, 10))

            # Show start hint
            if not started:
                draw_text("Press arrow keys or WASD to move", font, GRAY_COLOR, (SCREEN_W // 2, SCREEN_H - 30))

        pygame.display.flip()
        clock.tick(speed)


# Main
while True:
    result = game_loop()
    if result != "restart":
        break

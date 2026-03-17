import pygame

from nord_colors import nord0, nord4

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
from nord_colors import nord4, nord0

WHITE = nord4
BLACK = nord0

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong")

# Paddle class
class Paddle(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, color):
        super().__init__()
        self.image = pygame.Surface([width, height])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed = 5

    def update(self, up, down):
        if up:
            self.rect.y -= self.speed
        if down:
            self.rect.y += self.speed
        # Keep paddle within screen bounds
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > HEIGHT:
            self.rect.bottom = HEIGHT


# Ball class
class Ball(pygame.sprite.Sprite):
    def __init__(self, x, y, size, color):
        super().__init__()
        self.image = pygame.Surface([size, size])
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
        self.speed_x = 5
        self.speed_y = 5

    def update(self):
        self.rect.x += self.speed_x
        self.rect.y += self.speed_y
        # Bounce off top and bottom
        if self.rect.top < 0 or self.rect.bottom > HEIGHT:
            self.speed_y *= -1


# Paddle dimensions
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 100

# Create paddles
player1 = Paddle(50, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT, WHITE)
player2 = Paddle(WIDTH - 50 - PADDLE_WIDTH, HEIGHT // 2 - PADDLE_HEIGHT // 2, PADDLE_WIDTH, PADDLE_HEIGHT, WHITE)

# Ball dimensions
BALL_SIZE = 15

# Create ball
ball = Ball(WIDTH // 2 - BALL_SIZE // 2, HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, WHITE)

# Sprite groups
all_sprites = pygame.sprite.Group()
all_sprites.add(player1)
all_sprites.add(player2)
all_sprites.add(ball)

# Game loop
running = True
clock = pygame.time.Clock()
player1_up = False
player1_down = False
player2_up = False
player2_down = False

while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                player1_up = True
            elif event.key == pygame.K_s:
                player1_down = True
            elif event.key == pygame.K_UP:
                player2_up = True
            elif event.key == pygame.K_DOWN:
                player2_down = True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_w:
                player1_up = False
            elif event.key == pygame.K_s:
                player1_down = False
            elif event.key == pygame.K_UP:
                player2_up = False
            elif event.key == pygame.K_DOWN:
                player2_down = False

    # Update game state
    player1.update(player1_up, player1_down)
    player2.update(player2_up, player2_down)
    ball.update()

    # Collision detection
    if pygame.sprite.collide_rect(ball, player1) or pygame.sprite.collide_rect(ball, player2):
        ball.speed_x *= -1

    # Keep ball within screen bounds (left/right - game over)
    if ball.rect.left < 0 or ball.rect.right > WIDTH:
        print("Game Over!")
        running = False

    # Render
    screen.fill(BLACK)
    all_sprites.draw(screen)
    pygame.display.flip()

    # Limit frame rate
    clock.tick(60)

# Quit Pygame
pygame.quit()

import random

# ... (previous code) ...

# Game variables
...

food_pos = [random.randrange(1, (width//10)) * 10,
                random.randrange(1, (height//10)) * 10]
food_spawn = True

...


# Initialize Pygame
pygame.init()

# Window dimensions
width, height = 640, 480

# Colors (Dark theme)
black = (0, 0, 0)
white = (255, 255, 255)
green = (0, 255, 0)
red = (255, 0, 0)
dark_green = (0, 100, 0)

# Create the game window
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Snake Game")

# Clock object to control frame rate
clock = pygame.time.Clock()

# Game variables
snake_pos = [100, 50]
snake_body = [[100, 50],
              [90, 50],
              [80, 50]]

    # Game loop
    running = True
    while running and not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Handle key presses
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP or event.key == pygame.K_w:
                    change_to = "UP"
                if event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    change_to = "DOWN"
                if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    change_to = "LEFT"
                if event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    change_to = "RIGHT"

            # Validate direction
            if change_to == "UP" and direction != "DOWN":
                direction = "UP"
            if change_to == "DOWN" and direction != "UP":
                direction = "DOWN"
            if change_to == "LEFT" and direction != "RIGHT":
                direction = "LEFT"
            if change_to == "RIGHT" and direction != "LEFT":
                direction = "RIGHT"

            # Move the snake
            if direction == "UP":
                snake_pos[1] -= 10
            if direction == "DOWN":
                snake_pos[1] += 10
            if direction == "LEFT":
                snake_pos[0] -= 10
            if direction == "RIGHT":
                snake_pos[0] += 10

            # Snake body mechanism
            snake_body.insert(0, list(snake_pos))
            if snake_pos[0] == food_pos[0] and snake_pos[1] == food_pos[1]:
                score += 1
                food_spawn = False
            else:
                snake_body.pop()

            if not food_spawn:
                food_pos = [random.randrange(1, (width // 10)) * 10,
                            random.randrange(1, (height // 10)) * 10]
            food_spawn = True

            # Collision detection
            if snake_pos[0] < 0 or snake_pos[0] > width - 10:
                game_over = True
            if snake_pos[1] < 0 or snake_pos[1] > height - 10:
                game_over = True
            for block in snake_body[1:]:
                if snake_pos[0] == block[0] and snake_pos[1] == block[1]:
                    game_over = True

            # Increase snake speed
            game_speed = 10 + score // 5

            # Pygame screen updates
            screen.fill(black)
            for pos in snake_body:
                pygame.draw.rect(screen, green, pygame.Rect(pos[0], pos[1], 10, 10))

            pygame.draw.rect(screen, white, pygame.Rect(food_pos[0], food_pos[1], 10, 10))

            # Display score
            font = pygame.font.SysFont("consolas", 20)
            score_surface = font.render("Score : " + str(score), True, white)
            score_rect = score_surface.get_rect()
            screen.blit(score_surface, score_rect)

            # Game Over logic
            if game_over:
                font = pygame.font.SysFont("consolas", 50)
                game_over_surface = font.render("Game Over", True, red)
                game_over_rect = game_over_surface.get_rect()
                game_over_rect.midtop = (width / 2, height / 4)
                screen.blit(game_over_surface, game_over_rect)
                pygame.display.flip()
                while True:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            exit()
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                game_over = False
                                snake_pos = [100, 50]
                                snake_body = [[100, 50],
                                              [90, 50],
                                              [80, 50]]
                                food_pos = [random.randrange(1, (width // 10)) * 10,
                                            random.randrange(1, (height // 10)) * 10]
                                food_spawn = True
                                direction = "RIGHT"
                                change_to = direction
                                score = 0

            pygame.display.flip()
            clock.tick(game_speed)

    pygame.quit()
    exit()

direction = "RIGHT"
change_to = direction



              # Validate direction
            if change_to == "UP" and direction != "DOWN":
                direction = "UP"
            if change_to == "DOWN" and direction != "UP":
                direction = "DOWN"
            if change_to == "LEFT" and direction != "RIGHT":
                direction = "LEFT"
            if change_to == "RIGHT" and direction != "LEFT":
                direction = "RIGHT"

            # Move the snake
            if direction == "UP":
                snake_pos[1] -= 10
            if direction == "DOWN":
                snake_pos[1] += 10
            if direction == "LEFT":
                snake_pos[0] -= 10
            if direction == "RIGHT":
                snake_pos[0] += 10

    # Snake body mechanism
    # Snake body mechanism
    snake_body.insert(0, list(snake_pos))
    if snake_pos[0] == food_pos[0] and snake_pos[1] == food_pos[1]:
        score += 1
        food_spawn = False
    else:
        snake_body.pop()
    # Collision detection
    if snake_pos[0] < 0 or snake_pos[0] > width-10:
        game_over = True
    if snake_pos[1] < 0 or snake_pos[1] > height-10:
        game_over = True
    # Increase snake speed
    game_speed = 10 + score // 5

    # Pygame screen updates
    screen.fill(black)
    for pos in snake_body:
        pygame.draw.rect(screen, green, pygame.Rect(pos[0], pos[1], 10, 10))

    pygame.draw.rect(screen, white, pygame.Rect(food_pos[0], food_pos[1], 10, 10))

    # Display score
    font = pygame.font.SysFont("consolas", 20)
    score_surface = font.render("Score : " + str(score), True, white)
    score_rect = score_surface.get_rect()
    screen.blit(score_surface, score_rect)

    # Game Over logic
    if game_over:
        font = pygame.font.SysFont("consolas", 50)
        game_over_surface = font.render("Game Over", True, red)
        game_over_rect = game_over_surface.get_rect()
        game_over_rect.midtop = (width/2, height/4)
        screen.blit(game_over_surface, game_over_rect)
        pygame.display.flip()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                        pygame.quit()
        exit()
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        game_over = False
                        snake_pos = [100, 50]
                        snake_body = [[100, 50],
                                      [90, 50],
                                      [80, 50]]
                        food_pos = [random.randrange(1, (width//10)) * 10,
                                        random.randrange(1, (height//10)) * 10]
                        food_spawn = True
                        direction = "RIGHT"
                        change_to = direction
                        score = 0

    pygame.display.flip()
    clock.tick(game_speed)



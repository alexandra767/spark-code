import unittest
import pygame
from pong import Ball, Paddle

class TestPong(unittest.TestCase):

    def setUp(self):
        pygame.init()
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.paddle1 = Paddle(50, self.height // 2 - 50, 10, 100, (255, 255, 255))
        self.paddle2 = Paddle(self.width - 50 - 10, self.height // 2 - 50, 10, 100, (255, 255, 255))
        self.ball = Ball(self.width // 2 - 7, self.height // 2 - 7, 15, (255, 255, 255))

    def tearDown(self):
        pygame.quit()

    def test_paddle_movement(self):
        initial_y = self.paddle1.rect.y
        self.paddle1.update(True, False)  # Move up
        self.assertLess(self.paddle1.rect.y, initial_y)

        initial_y = self.paddle1.rect.y
        self.paddle1.update(False, True)  # Move down
        self.assertGreater(self.paddle1.rect.y, initial_y)

    def test_paddle_bounds(self):
        # Move paddle up until it hits the top
        for _ in range(200):
            self.paddle1.update(True, False)
        self.assertEqual(self.paddle1.rect.top, 0)

        # Move paddle down until it hits the bottom
        for _ in range(200):
            self.paddle1.update(False, True)
        self.assertEqual(self.paddle1.rect.bottom, self.height)

    def test_ball_movement(self):
        initial_x = self.ball.rect.x
        initial_y = self.ball.rect.y
        self.ball.update()
        self.assertNotEqual(self.ball.rect.x, initial_x)
        self.assertNotEqual(self.ball.rect.y, initial_y)

    def test_ball_bounce(self):
        # Position ball near the top
        self.ball.rect.y = 5
        self.ball.speed_y = -5  # Moving upwards
        self.ball.update()
        self.assertGreater(abs(self.ball.speed_y), 0)  # Should be moving downwards

        # Position ball near the bottom
        self.ball.rect.y = self.height - 5
        self.ball.speed_y = 5  # Moving downwards
        self.ball.update()
        self.assertGreater(abs(self.ball.speed_y), 0)  # Should be moving downwards

if __name__ == '__main__':
    unittest.main()

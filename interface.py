import pygame
from queue import Queue, Empty

# --------------------------------
# "Interface" : ton BCI enverra ici
# --------------------------------
actions = Queue()

def push_action(action: str):
    """À remplacer plus tard par ton module EEG."""
    actions.put(action)

# --------------------------------
# Mini programme
# --------------------------------
pygame.init()
W, H = 800, 450
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("BCI Mini Program (MVP)")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)

x = W // 2
y = H // 2
speed = 14
running = True
enabled = True  # toggle start/stop
last_action = "NONE"

while running:
    # 1) Input (clavier pour simuler)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
                push_action("LEFT")
            elif event.key == pygame.K_d:
                push_action("RIGHT")
            elif event.key == pygame.K_SPACE:
                push_action("TOGGLE")
            elif event.key == pygame.K_s:
                push_action("STOP")

    # 2) Consommer les actions (non bloquant)
    while True:
        try:
            action = actions.get_nowait()
        except Empty:
            break

        last_action = action

        if action == "TOGGLE":
            enabled = not enabled
        elif action == "STOP":
            enabled = False
        elif enabled and action == "LEFT":
            x -= speed
        elif enabled and action == "RIGHT":
            x += speed

    # 3) Clamp
    x = max(40, min(W - 40, x))

    # 4) Render
    screen.fill((20, 20, 25))
    pygame.draw.rect(screen, (240, 240, 240), pygame.Rect(x - 30, y - 30, 60, 60))

    status = f"Enabled: {enabled} | Last action: {last_action} | Keys: A=LEFT D=RIGHT SPACE=TOGGLE S=STOP"
    screen.blit(font.render(status, True, (200, 200, 200)), (20, 20))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
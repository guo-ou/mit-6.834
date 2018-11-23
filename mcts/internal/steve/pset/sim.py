import game
import algo
import time

from IPython.display import display, display_html, display_markdown, IFrame



def simulate_game(uct, visualize=False, json_vis=False):
#    make_game_vis()
    print("start simulation")
    time_limit_1 = 0.4
    time_limit_2 = 0.4
    board = game.ConnectFourBoard()
    player_1 = game.ComputerPlayer('mcts', uct, time_limit_1)
    player_2 = game.ComputerPlayer('alpha-beta', algo.alpha_beta_algo, time_limit_2)
    sim = game.Simulation(board, player_1, player_2)
    sim.run(visualize, json_vis)
#    time.sleep(0.1)
    print("Player", sim.board.current_player_id()," won")
    return sim.board.current_player_id()

def make_game_vis():
    frame = IFrame('vis/index.html', 490, 216)
    display(frame)

def run_final_test(uct, n):
    losses = 0
    for i in range(n):
        loser = simulate_game(uct)
        if loser == 0:
            losses += 1
            if losses > 1:
                lose()
                return
    win()

def win():
    print("win")
    display_html("""<div class="alert alert-success">
    <strong>You win!!</strong></div>""", raw=True)

def lose():
    print("lose")
    display_html("""<div class="alert alert-failure">
    <strong>You can only lose once :(</strong>
    </div>""", raw=True)


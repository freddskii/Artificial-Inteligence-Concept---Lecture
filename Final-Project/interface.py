
"""
ACOnGA_2_tui_launcher.py

A non-destructive TUI launcher that wraps the existing /mnt/data/ACOnGA_2.py
- Provides an ncurses dashboard menu
- Reuses existing visualization functions by suspending curses and opening their GUI windows
- Adds a Batch "ACO vs GA" runner that executes N rounds for S seconds per round
- Does NOT modify the original ACO/GA logic. It dynamically imports the uploaded file.

Usage: python ACOnGA_2_tui_launcher.py

The launcher will import the user's uploaded ACOnGA_2.py from /mnt/data/ACOnGA_2.py
"""

import importlib.util
import sys
import os
import time
import random
import curses
import numpy as np
from types import ModuleType

MODULE_PATH = './ACOnGA_2.py'  # uploaded file (do not change)


def load_module(path: str) -> ModuleType:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Module file not found: {path}")
    spec = importlib.util.spec_from_file_location('aconga2_uploaded', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


class TUILauncher:
    def __init__(self, module: ModuleType):
        self.mod = module
        # Expect ACOvsGAController and InteractivePlacementGUI to exist in the module
        self.ControllerClass = getattr(self.mod, 'ACOvsGAController', None)
        self.PlacementGUI = getattr(self.mod, 'InteractivePlacementGUI', None)
        self.GraphicalPlacement = getattr(self.mod, 'GraphicalDronePlacement', None)

        if not self.ControllerClass:
            raise RuntimeError('ACOvsGAController not found in module')

        self.controller = self.ControllerClass()

    # Utility: suspend curses, run func that interacts with terminal / opens GUI, then resume curses
    def suspend_curses_and_run(self, func, *args, **kwargs):
        # End curses window to allow matplotlib/GUI windows or blocking input
        try:
            curses.endwin()
        except Exception:
            pass
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            print(f"Error while running function: {e}")
            import traceback
            traceback.print_exc()
            result = None
        input('\nPress ENTER to return to the TUI...')
        # Reinitialize curses will be handled by caller (we simply return)
        return result

    # MENU ACTIONS
    def action_initialize_random_swarm(self):
        def work():
            num = int(input('Enter number of drones to initialize (recommended 15-30, default 20): ') or '20')
            self.controller.initialize_swarm(num)
            self.controller.update_neighbor_relationships()
            print(f'Initialized swarm with {len(self.controller.drones)} drones.')
        return self.suspend_curses_and_run(work)

    def action_graphical_placement(self):
        # Use existing graphical placement GUI if available
        if self.PlacementGUI:
            def work():
                gui = self.PlacementGUI(area_size=(2000, 2000))
                completed = gui.start_placement_interface()
                if completed:
                    # Replace controller with the one used in the GUI if available
                    if hasattr(gui, 'controller'):
                        self.controller = gui.controller
                        print('Placement complete. Controller loaded from GUI.')
                    else:
                        print('Placement finished but no controller returned by GUI.')
                else:
                    print('Placement cancelled.')
            return self.suspend_curses_and_run(work)
        elif self.GraphicalPlacement:
            def work():
                gui = self.GraphicalPlacement(area_size=(2000, 2000))
                completed = gui.start_placement_interface()
                if completed and hasattr(gui, 'controller'):
                    self.controller = gui.controller
                    print('Placement complete. Controller loaded from graphical placement.')
            return self.suspend_curses_and_run(work)
        else:
            print('\nGraphical placement GUI not available in the module.')

    def action_show_network_overview(self):
        if hasattr(self.controller, 'show_network_overview'):
            return self.suspend_curses_and_run(self.controller.show_network_overview)
        else:
            print('show_network_overview not found on controller')

    def action_view_network_static(self):
        # call visualize_network_static or visualize_network_static (different names in files)
        fn = None
        for name in ('visualize_network_static', 'visualize_network_static', 'visualize_network_static'):
            if hasattr(self.controller, name):
                fn = getattr(self.controller, name)
                break
        # fall back to controller.visualize_network_static
        if not fn and hasattr(self.controller, 'visualize_network_static'):
            fn = getattr(self.controller, 'visualize_network_static')

        if fn:
            return self.suspend_curses_and_run(fn)
        else:
            print('No network static visualization function found on controller')

    def action_run_aco_once(self):
        # Interactive arrow-key selection for source/destination
        def curses_select_node(title, stdscr, ids):
            idx = 0
            while True:
                stdscr.clear()
                stdscr.addstr(1, 2, title, curses.A_BOLD)
                for i, node in enumerate(ids):
                    if i == idx:
                        stdscr.attron(curses.color_pair(1))
                        stdscr.addstr(3+i, 4, f"> {node}")
                        stdscr.attroff(curses.color_pair(1))
                    else:
                        stdscr.addstr(3+i, 6, node)
                key = stdscr.getch()
                if key in (curses.KEY_UP, ord('k')):
                    idx = (idx - 1) % len(ids)
                elif key in (curses.KEY_DOWN, ord('j')):
                    idx = (idx + 1) % len(ids)
                elif key in (curses.KEY_ENTER, ord('\n'), ord('\r')):
                    return ids[idx]
            return ids[idx]

        def work():
            ids = sorted(self.controller.drones.keys())
            if len(ids) < 2:
                print('Need at least 2 drones to run ACO.')
                return
            # Use curses wrapper for interactive selection
            source = curses.wrapper(lambda s: curses_select_node('Select SOURCE node', s, ids))
            dest_list = [i for i in ids if i != source]
            dest = curses.wrapper(lambda s: curses_select_node('Select DESTINATION node', s, dest_list))
            duration = int(input('Duration (seconds, default 10): ') or '10')
            reset = input('Reset pheromones before run? (y/N): ').strip().lower() == 'y'
            print(f'Running ACO round {source} -> {dest} for {duration}s (reset={reset})')

            fn = getattr(self.controller, 'run_aco_only_test', None)
            if callable(fn):
                res = fn(source, dest)
                self.controller.visualize_route(res[-1], "ACO", source, dest)

        return self.suspend_curses_and_run(work)

    def action_run_ga_once(self):
        def curses_select_node(title, stdscr, ids):
            idx = 0
            while True:
                stdscr.clear()
                stdscr.addstr(1, 2, title, curses.A_BOLD)
                for i, node in enumerate(ids):
                    if i == idx:
                        stdscr.attron(curses.color_pair(1))
                        stdscr.addstr(3+i, 4, f"> {node}")
                        stdscr.attroff(curses.color_pair(1))
                    else:
                        stdscr.addstr(3+i, 6, node)
                key = stdscr.getch()
                if key in (curses.KEY_UP, ord('k')):
                    idx = (idx - 1) % len(ids)
                elif key in (curses.KEY_DOWN, ord('j')):
                    idx = (idx + 1) % len(ids)
                elif key in (curses.KEY_ENTER, ord('\n'), ord('\r')):
                    return ids[idx]
            return ids[idx]

        def work():
            ids = sorted(self.controller.drones.keys())
            if len(ids) < 2:
                print('Need at least 2 drones to run ACO.')
                return
            # Use curses wrapper for interactive selection
            source = curses.wrapper(lambda s: curses_select_node('Select SOURCE node', s, ids))
            dest_list = [i for i in ids if i != source]
            dest = curses.wrapper(lambda s: curses_select_node('Select DESTINATION node', s, dest_list))
            if source == dest:
                print('Source and destination must differ.')
                return
            print(f'Running GA test {source} -> {dest}')
            fn = getattr(self.controller, 'run_ga_only_test', None)
            if callable(fn):
                res = fn(source, dest)
                self.controller.visualize_route(res[-1], "GA", source, dest)



        return self.suspend_curses_and_run(work)

    def action_batch_aco_vs_ga(self):
        def work():
            seconds = int(input('Enter number of seconds to simulate (default 30): ').strip() or '30')
            ga_repeats = int(input('GA comparison repeats per round (default 5): ').strip() or '5')

            ids = sorted(self.controller.drones.keys())
            if len(ids) < 2:
                print('Need at least 2 drones to run batch test.')
                return

            print(f'Running batch test: {seconds} seconds, GA repeats={ga_repeats}')

            if(hasattr(self.controller, "run_simulation")): 
                print(f"Running Simulation with {seconds} seconds of total time")
                self.controller.run_simulation(seconds, ga_repeats)


        return self.suspend_curses_and_run(work)

    # TUI main loop
    def run_curses(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)

        menu = [
            'Initialize random swarm',
            'Graphical placement (opens GUI window)',
            'Show network overview (opens GUI/prints)',
            'View network (static visualization)',
            'Run ACO once',
            'Run GA once',
            'Run ACO vs GA batch test',
            'Exit'
        ]

        actions = [
            self.action_initialize_random_swarm,
            self.action_graphical_placement,
            self.action_show_network_overview,
            self.action_view_network_static,
            self.action_run_aco_once,
            self.action_run_ga_once,
            self.action_batch_aco_vs_ga,
            None
        ]

        current = 0
        while True:
            stdscr.clear()
            h, w = stdscr.getmaxyx()
            title = 'ACO vs GA - Terminal Dashboard'
            stdscr.addstr(1, w//2 - len(title)//2, title, curses.A_BOLD)
            stdscr.addstr(3, 2, f'Network drones: {len(self.controller.drones)}')
            stdscr.addstr(4, 2, f'Rounds completed: {len(getattr(self.controller, "round_history", []))}')
            stdscr.addstr(5, 2, 'Use UP/DOWN to navigate, ENTER to select')

            for idx, item in enumerate(menu):
                x = 8 + idx
                if idx == current:
                    stdscr.attron(curses.color_pair(1))
                    stdscr.addstr(x, 4, f'> {item}')
                    stdscr.attroff(curses.color_pair(1))
                else:
                    stdscr.addstr(x, 6, item)

            key = stdscr.getch()
            if key in (curses.KEY_UP, ord('k')):
                current = (current - 1) % len(menu)
            elif key in (curses.KEY_DOWN, ord('j')):
                current = (current + 1) % len(menu)
            elif key in (curses.KEY_ENTER, ord('\n'), ord('\r')):
                if menu[current] == 'Exit':
                    break
                action = actions[current]
                if(len(self.controller.drones) <= 0 and not(menu[current] == 'Graphical placement (opens GUI window)' or menu[current] == 'Initialize random swarm')): 
                    continue
                if action:
                    # execute action; action returns after user presses enter to continue
                    # reinit curses screen after action
                    action()
                    # reinitialize curses screen state
                    stdscr = curses.initscr()
                    curses.noecho(); curses.cbreak(); stdscr.keypad(True)
                else:
                    pass
            elif key in (ord('q'), ord('Q')):
                break
            stdscr.refresh()
            time.sleep(0.05)


def main():
    mod = load_module(MODULE_PATH)
    launcher = TUILauncher(mod)

    # initialize curses color pair
    try:
        curses.wrapper(lambda scr: (curses.start_color(), curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN), launcher.run_curses(scr)))
    except Exception as e:
        print('Error running TUI:', e)
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()


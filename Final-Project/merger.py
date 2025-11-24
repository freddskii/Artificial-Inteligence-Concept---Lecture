
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

MODULE_PATH = './ACOnGA.py'  # uploaded file (do not change)


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

    def action_show_pheromone_map(self):
        if not hasattr(self.controller, 'show_pheromone_map'):
            print('show_pheromone_map not available on controller')
            return
        def work():
            dest = input('Enter destination drone ID for pheromone map (or index): ').strip()
            # allow index to id mapping
            if dest.isdigit():
                idx = int(dest) - 1
                ids = sorted(self.controller.drones.keys())
                if 0 <= idx < len(ids):
                    dest = ids[idx]
                else:
                    print('Invalid index')
                    return
            if dest not in self.controller.drones:
                print('Destination ID not found in current network')
                return
            self.controller.show_pheromone_map(dest)
        return self.suspend_curses_and_run(work)

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
            if hasattr(self.controller, 'run_round'):
                self.controller.run_round(source, dest, duration, reset_pheromones=reset)
            else:
                fn = getattr(self.controller, 'test_aco_routing', None)
                if callable(fn):
                    res = fn(source, dest)
                    print('ACO result:', res)
                # show pheromone map after single test
                if hasattr(self.controller, 'show_pheromone_map'):
                    print('Opening pheromone map for destination...')
                    self.controller.show_pheromone_map(dest)
                else:
                    print('No ACO run function found')
            # After successful run of run_round
            if hasattr(self.controller, 'show_pheromone_map'):
                print('Opening pheromone map for destination...')
                self.controller.show_pheromone_map(dest)
        return self.suspend_curses_and_run(work)

    def action_run_ga_once(self):
        def work():
            ids = sorted(self.controller.drones.keys())
            if len(ids) < 2:
                print('Need at least 2 drones to run GA.')
                return
            s = input('Source drone (id or index, blank=random): ').strip()
            if not s:
                source = random.choice(ids)
            elif s.isdigit():
                idx = int(s) - 1
                source = ids[idx]
            else:
                source = s
            d = input('Destination drone (id or index, blank=random): ').strip()
            if not d:
                dest = random.choice([i for i in ids if i != source])
            elif d.isdigit():
                idx = int(d) - 1
                dest = ids[idx]
            else:
                dest = d
            if source == dest:
                print('Source and destination must differ.')
                return
            print(f'Running GA test {source} -> {dest}')
            fn = getattr(self.controller, 'test_ga_routing', None)
            if callable(fn):
                res = fn(source, dest)
                print('GA result:', res)
            else:
                print('GA test function not found on controller')
        return self.suspend_curses_and_run(work)

    def action_batch_aco_vs_ga(self):
        def work():
            rounds = int(input('Enter number of rounds (N): ').strip() or '5')
            seconds_per_round = int(input('Seconds per round: ').strip() or '10')
            reset_choice = input('Reset pheromones at each round? (y/N): ').strip().lower() == 'y'
            # How many GA instantaneous comparison runs per round
            ga_repeats = int(input('GA comparison repeats per round (default 5): ').strip() or '5')

            ids = sorted(self.controller.drones.keys())
            if len(ids) < 2:
                print('Need at least 2 drones to run batch test.')
                return

            print(f'Running batch test: {rounds} rounds, {seconds_per_round}s per round, reset_each={reset_choice}, GA reps={ga_repeats}')

            # Ensure metrics containers exist
            if not hasattr(self.controller, 'batch_stats_aco'):
                self.controller.batch_stats_aco = []
            if not hasattr(self.controller, 'batch_stats_ga'):
                self.controller.batch_stats_ga = []

            for r in range(1, rounds + 1):
                source = random.choice(ids)
                dest = random.choice([i for i in ids if i != source])
                print(f'--- Round {r}/{rounds} : {source} -> {dest} ---')

                # Run the existing ACO round (this performs multiple ants over duration)
                if hasattr(self.controller, 'run_round'):
                    self.controller.run_round(source, dest, seconds_per_round, reset_pheromones=reset_choice)
                else:
                    print('No run_round found; skipping ACO execution for this round')

                # Collect ACO round summary from controller.round_history (last entry)
                aco_summary = None
                if hasattr(self.controller, 'round_history') and self.controller.round_history:
                    aco_summary = self.controller.round_history[-1]
                    self.controller.batch_stats_aco.append(aco_summary)

                # Run GA instantaneous comparisons ga_repeats times (user chose GA to run instantly)
                ga_results = []
                for g_idx in range(ga_repeats):
                    ga_res = None
                    if hasattr(self.controller, 'test_ga_routing'):
                        ga_res = self.controller.test_ga_routing(source, dest)
                        ga_results.append(ga_res)

                        # Update GA metrics (count every attempt; failure counts as failed)
                        if not hasattr(self.controller, 'ga_metrics'):
                            self.controller.ga_metrics = {'routes_found':0,'routes_failed':0,'total_hops':[],'latencies':[],'path_qualities':[],'success_rate':[]}

                        if ga_res.success:
                            self.controller.ga_metrics['routes_found'] += 1
                            self.controller.ga_metrics['total_hops'].append(ga_res.hop_count)
                            self.controller.ga_metrics['latencies'].append(ga_res.latency)
                            self.controller.ga_metrics['path_qualities'].append(ga_res.path_quality)
                        else:
                            self.controller.ga_metrics['routes_failed'] += 1
                    else:
                        print('GA test function not found; skipping GA run')

                # Store GA aggregated info for this round
                if ga_results:
                    # simple aggregation per round
                    success_count = sum(1 for x in ga_results if x.success)
                    fail_count = len(ga_results) - success_count
                    avg_quality = float(np.mean([x.path_quality for x in ga_results if x.success])) if any(x.success for x in ga_results) else 0.0
                    self.controller.batch_stats_ga.append({
                        'round': r,
                        'source': source,
                        'destination': dest,
                        'ga_attempts': len(ga_results),
                        'ga_successful': success_count,
                        'ga_failed': fail_count,
                        'ga_avg_quality': avg_quality
                    })

                # Print quick per-round GA summary
                if ga_results:
                    print(f"GA: {success_count} successful, {fail_count} failed (of {len(ga_results)} attempts); avg quality: {avg_quality*100:.1f}%")

            # After batch, print combined summary using existing functions where possible
            print('Batch complete. Printing final comparison:')

            # If controller has print_final_comparison, call it to print any previously gathered metrics
            if hasattr(self.controller, 'print_final_comparison'):
                self.controller.print_final_comparison()

            # Print aggregated GA batch stats summary
            if hasattr(self.controller, 'batch_stats_ga') and self.controller.batch_stats_ga:
                total_ga_attempts = sum(x['ga_attempts'] for x in self.controller.batch_stats_ga)
                total_ga_success = sum(x['ga_successful'] for x in self.controller.batch_stats_ga)
                total_ga_failed = sum(x['ga_failed'] for x in self.controller.batch_stats_ga)
                avg_ga_quality = float(np.mean([x['ga_avg_quality'] for x in self.controller.batch_stats_ga if x['ga_avg_quality']>0])) if any(x['ga_avg_quality']>0 for x in self.controller.batch_stats_ga) else 0.0
                print('GA Batch Summary:')
                print(f'  Total GA attempts: {total_ga_attempts}')
                print(f'  Successful GA routes: {total_ga_success}')
                print(f'  Failed GA routes: {total_ga_failed}')
                if total_ga_attempts > 0:
                    print(f'  GA overall success rate: {total_ga_success/total_ga_attempts*100:.1f}%')
                if avg_ga_quality > 0:
                    print(f'  GA average path quality: {avg_ga_quality*100:.1f}%')

            # Also print round history (ACO rounds)
            if hasattr(self.controller, 'round_history'):
                try:
                    self.mod.print_round_history(self.controller)
                except Exception:
                    pass
                # Or controller-level printer
                if hasattr(self.controller, 'print_round_history'):
                    try:
                        self.controller.print_round_history()
                    except Exception:
                        pass
        return self.suspend_curses_and_run(work)

    # TUI main loop
    def run_curses(self, stdscr):
        max_y, max_x = stdscr.getmaxyx()
        curses.curs_set(0)
        stdscr.nodelay(False)
        stdscr.keypad(True)

        menu = [
            'Initialize random swarm',
            'Graphical placement (opens GUI window)',
            'Show network overview (opens GUI/prints)',
            'View network (static visualization)',
            'Show pheromone map (opens GUI)',
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
            self.action_show_pheromone_map,
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
        if x < max_y - 1:  # <--- Add this check
            try:
                # Ensure string isn't wider than the screen either
                display_text = item[:max_x - 7] 
                stdscr.addstr(x, 6, display_text)
            except curses.error:
                pass


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

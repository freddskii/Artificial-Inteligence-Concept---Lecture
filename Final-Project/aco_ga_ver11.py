import math
import random
import time
import threading
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as mpatches

RNG = 4136314
random.seed(RNG)
np.random.seed(RNG)

class DroneType(Enum):
    LEADER = "leader"
    WORKER = "worker"

class AntType(Enum):
    EXPLORATORY = "exploratory"
    DATA_COLLECTION = "data_collection"
    EMERGENCY = "emergency"

@dataclass
class Position:
    x: float
    y: float
    z: float
    
    def distance_to(self, other: 'Position') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

@dataclass
class LinkMetrics:
    rssi: float
    latency: float
    packet_loss: float
    bandwidth: float
    last_updated: float

# ============= ACO COMPONENTS =============

class ForwardAnt:
    def __init__(self, ant_id: str, source_drone: 'UnifiedDrone', destination_drone: 'UnifiedDrone', 
                 ant_type: AntType = AntType.EXPLORATORY):
        self.ant_id = ant_id
        self.source_drone = source_drone
        self.destination_drone = destination_drone
        self.ant_type = ant_type
        self.path_taken = []
        self.quality_metrics = {}
        self.hop_count = 0
        self.timestamp = time.time()
        self.max_hops = 20
        self.ttl = 30.0
        self.emergency_flag = False
        
    def record_hop(self, drone_id: str, link_quality: float, position: Position):
        self.path_taken.append({
            'drone_id': drone_id,
            'position': position,
            'link_quality': link_quality,
            'timestamp': time.time()
        })
        self.hop_count += 1
        
    def should_continue(self) -> bool:
        if self.hop_count >= self.max_hops:
            return False
        if time.time() - self.timestamp > self.ttl:
            return False
        if self.path_taken and self.path_taken[-1]['drone_id'] == self.destination_drone.drone_id:
            return False
        return True
    
    def calculate_current_path_quality(self) -> float:
        if not self.path_taken:
            return 0.0
        total_quality = sum(hop['link_quality'] for hop in self.path_taken)
        return total_quality / len(self.path_taken)

class BackwardAnt:
    def __init__(self, forward_ant: ForwardAnt):
        self.forward_ant = forward_ant
        self.path_quality_score = 0.0
        self.pheromone_updates = {}
        self.emergency_flag = forward_ant.emergency_flag
        self.timestamp = time.time()
        
    def calculate_path_metrics(self) -> Dict[str, float]:
        # --- THIS IS THE FIX ---
        # We must skip the first hop (the source node) in our calculation,
        # because its link_quality is always a meaningless default of 1.0.
        # We only care about the quality of the *links* we traversed.
        
        path_links = self.forward_ant.path_taken[1:] # Slice to skip hop 0
        
        if not path_links:
            # The path was just the source node (ant failed immediately)
            return {}
            
        total_quality = sum(hop['link_quality'] for hop in path_links)
        avg_quality = total_quality / len(path_links) # Divide by the number of links
        
        # Hops are (nodes - 1), which is len(path_links)
        hop_count = len(path_links) 
        hop_penalty = hop_count * 0.1
        # --- END OF FIX ---
        
        if self.emergency_flag:
            avg_quality *= 1.2
        
        self.path_quality_score = avg_quality - hop_penalty
        return {
            'quality_score': self.path_quality_score,
            'hop_count': hop_count,
            'avg_quality': avg_quality
        }
        
    def update_pheromones(self, drone_network: Dict[str, 'UnifiedDrone']):
        metrics = self.calculate_path_metrics()
        path_quality = max(0.1, metrics['quality_score'])
        destination_id = self.forward_ant.destination_drone.drone_id
        
        for i in range(len(self.forward_ant.path_taken) - 1):
            current_drone_id = self.forward_ant.path_taken[i]['drone_id']
            next_drone_id = self.forward_ant.path_taken[i + 1]['drone_id']
            
            if current_drone_id in drone_network and next_drone_id in drone_network:
                current_drone = drone_network[current_drone_id]
                current_drone.update_pheromone_aco(destination_id, next_drone_id, path_quality)
                
                if path_quality > 0.5:
                    current_drone.routing_table_aco[self.forward_ant.destination_drone.drone_id] = next_drone_id

# ============= UNIFIED DRONE CLASS =============

class UnifiedDrone:
    def __init__(self, drone_id: str, position: Position, drone_type: DroneType, 
                 initial_energy: float = 100.0):
        self.drone_id = drone_id
        self.position = position
        self.drone_type = drone_type
        self.battery_level = initial_energy
        self.initial_energy = initial_energy
        self.communication_range = 2000.0 if drone_type == DroneType.LEADER else 1000.0
        
        self.neighbor_drones = {}
        
        # ACO components
        self.pheromone_table_aco = {}
        self.routing_table_aco = {}
        self.ants_processed = 0
        
        self.packets_forwarded_aco = 0
        self.energy_used = 0.0
        
        self.lock = threading.RLock()
        
    def update_position(self, new_position: Position):
        distance_moved = self.position.distance_to(new_position)
        movement_energy = distance_moved * 0.01
        
        with self.lock:
            self.position = new_position
            self.battery_level -= movement_energy
            self.energy_used += movement_energy
            
    def measure_link_quality(self, neighbor_drone: 'UnifiedDrone') -> float:
        distance = self.position.distance_to(neighbor_drone.position)
        
        if distance > self.communication_range:
            return 0.0
            
        los_quality = 1.0 - (distance / self.communication_range) ** 2
        
        if self.drone_type == DroneType.LEADER or neighbor_drone.drone_type == DroneType.LEADER:
            signal_boost = 1.2
        else:
            signal_boost = 1.0
            
        energy_factor = min(self.battery_level, neighbor_drone.battery_level) / 100.0
        
        link_quality = los_quality * signal_boost * energy_factor
        return max(0.0, min(1.0, link_quality))
    
    def calculate_heuristic(self, neighbor_drone: 'UnifiedDrone', destination: 'UnifiedDrone') -> float:
        los_quality = self.measure_link_quality(neighbor_drone)
        
        distance_to_neighbor = self.position.distance_to(neighbor_drone.position)
        signal_strength = 1.0 - (distance_to_neighbor / self.communication_range)
        
        battery_compatibility = min(self.battery_level, neighbor_drone.battery_level) / 100.0
        
        current_to_dest = self.position.distance_to(destination.position)
        neighbor_to_dest = neighbor_drone.position.distance_to(destination.position)
        progress = 1.0 if neighbor_to_dest < current_to_dest else 0.5
        
        heuristic = (0.30 * los_quality + 
                    0.25 * signal_strength + 
                    0.20 * battery_compatibility + 
                    0.25 * progress)
        
        return max(0.1, heuristic)
    
    # ===== ACO METHODS =====
    
    def process_forward_ant_aco(self, ant: ForwardAnt, drone_network: Dict[str, 'UnifiedDrone']) -> Optional[str]:
        with self.lock:
            self.ants_processed += 1
            
            # === THIS IS THE FIX ===
            # Calculate the real link quality of the hop we just made
            # instead of just 'link_quality = 1.0'
            link_quality = 1.0  # Default for the first hop (the source drone)
            if ant.path_taken:
                # If we have a path, find the previous drone
                previous_drone_id = ant.path_taken[-1]['drone_id']
                if previous_drone_id in self.neighbor_drones:
                    # Get the quality of the link we just traversed
                    link_metrics = self.neighbor_drones[previous_drone_id]
                    link_quality = link_metrics.rssi / 100.0
                else:
                    # This node is no longer a neighbor (or was never), penalize
                    link_quality = 0.1 
            # === END OF FIX ===
            
            ant.record_hop(self.drone_id, link_quality, self.position)
            
            if self.drone_id == ant.destination_drone.drone_id:
                return None
                
            if not ant.should_continue():
                return None
                
            available_neighbors = self.get_available_neighbors_aco(ant)
            if not available_neighbors:
                return None
                
            # --- THIS CALL IS MODIFIED ---
            next_hop_id = self.advanced_routing_decision_aco(
                available_neighbors, ant.destination_drone, drone_network, ant.ant_type
            )
            # --- END MODIFICATION ---
            return next_hop_id
            
    def get_available_neighbors_aco(self, ant: ForwardAnt) -> List[str]:
        available = []
        visited_drones = {hop['drone_id'] for hop in ant.path_taken}
        
        for neighbor_id in self.neighbor_drones.keys():
            if neighbor_id not in visited_drones:
                available.append(neighbor_id)
                
        return available
        
    def advanced_routing_decision_aco(self, available_neighbors: List[str], 
                                     destination: 'UnifiedDrone',
                                     drone_network: Dict[str, 'UnifiedDrone'],
                                     ant_type: AntType = AntType.EXPLORATORY) -> str:
        if not available_neighbors:
            return None
            
        probabilities = []
        total = 0.0
        
        if ant_type == AntType.EMERGENCY:
            alpha = 0.4
            beta = 0.6
        elif ant_type == AntType.DATA_COLLECTION:
            alpha = 0.7
            beta = 0.3
        else:
            alpha = 0.6
            beta = 0.4
        
        for neighbor_id in available_neighbors:
            if destination.drone_id not in self.pheromone_table_aco:
                self.pheromone_table_aco[destination.drone_id] = {}
            if neighbor_id not in self.pheromone_table_aco[destination.drone_id]:
                self.pheromone_table_aco[destination.drone_id][neighbor_id] = 0.1
                
            pheromone = self.pheromone_table_aco[destination.drone_id][neighbor_id]
            
            # === THIS IS THE FIX ===
            # Calculate a real heuristic based on available link metrics
            heuristic = 0.1  # Default low value
            if neighbor_id in drone_network: # Use passed-in drone_network
                neighbor_drone = drone_network[neighbor_id] # Use passed-in drone_network
                # Use the actual heuristic calculation
                heuristic = self.calculate_heuristic(neighbor_drone, destination)
            # === END OF FIX ===

            probability = (pheromone ** alpha) * (heuristic ** beta)
            probabilities.append((neighbor_id, probability))
            total += probability
            
        if total > 0:
            random_value = random.uniform(0, total)
            cumulative = 0.0
            for neighbor_id, prob in probabilities:
                cumulative += prob
                if random_value <= cumulative:
                    return neighbor_id
                    
        return random.choice(available_neighbors)
    
    def update_pheromone_aco(self, destination_id: str, neighbor_id: str, path_quality: float):
        evaporation_rate = 0.3
        Q = 2.0
        
        with self.lock:
            if destination_id not in self.pheromone_table_aco:
                self.pheromone_table_aco[destination_id] = {}
            
            current_pheromone = self.pheromone_table_aco[destination_id].get(neighbor_id, 0.1)
            
            evaporated_pheromone = current_pheromone * (1 - evaporation_rate)
            
            reinforcement = Q * path_quality
            new_pheromone = evaporated_pheromone + reinforcement
            
            new_pheromone = max(0.1, min(1.0, new_pheromone))
            
            self.pheromone_table_aco[destination_id][neighbor_id] = new_pheromone
    
    def add_neighbor(self, neighbor_id: str, link_metrics: LinkMetrics):
        with self.lock:
            self.neighbor_drones[neighbor_id] = link_metrics
    
    def reset_pheromones(self):
        """Reset all pheromones to initial values"""
        with self.lock:
            self.pheromone_table_aco = {}
            self.routing_table_aco = {}

# ============= ACO CONTROLLER =============

class InteractiveACOController:
    def __init__(self):
        self.drones: Dict[str, UnifiedDrone] = {}
        self.running = False
        self.simulation_time = 0
        self.ant_counter = 0
        self.round_number = 0
        
        # ACO parameters
        self.aco_params = {
            'alpha': 0.6,
            'beta': 0.4,
            'rho': 0.3,
            'Q': 2.0,
            'ants_per_second': 2
        }
        
        # Round history
        self.round_history = []
        self.current_round_results = []
        
    def initialize_swarm(self, num_drones: int, area_size: Tuple[float, float, float] = (2000, 2000, 300)):
        print(f"Initializing swarm with {num_drones} drones...")
        
        num_leaders = max(1, num_drones // 10)
        for i in range(num_leaders):
            position = Position(
                x=random.uniform(0, area_size[0]),
                y=random.uniform(0, area_size[1]),
                z=random.uniform(100, area_size[2])
            )
            drone_id = f"leader_{i}"
            self.drones[drone_id] = UnifiedDrone(drone_id, position, DroneType.LEADER, 150.0)
        
        for i in range(num_drones - num_leaders):
            position = Position(
                x=random.uniform(0, area_size[0]),
                y=random.uniform(0, area_size[1]),
                z=random.uniform(50, area_size[2])
            )
            drone_id = f"worker_{i}"
            self.drones[drone_id] = UnifiedDrone(drone_id, position, DroneType.WORKER, 100.0)
        
        self.update_neighbor_relationships()
        print(f"Swarm initialized: {num_leaders} leaders, {num_drones - num_leaders} workers")

    def update_neighbor_relationships(self):
        drone_ids = list(self.drones.keys())
        
        # --- FIX: Clear all neighbor lists FIRST ---
        for drone_id in drone_ids:
            if drone_id in self.drones:
                self.drones[drone_id].neighbor_drones.clear()
        # --- END FIX ---
        
        for i, drone_id1 in enumerate(drone_ids):
            drone1 = self.drones[drone_id1]
            # The line 'drone1.neighbor_drones.clear()' was here and has been REMOVED
            
            for drone_id2 in drone_ids[i+1:]:
                drone2 = self.drones[drone_id2]
                distance = drone1.position.distance_to(drone2.position)
                
                # Check if EITHER drone can reach the other
                max_range = max(drone1.communication_range, drone2.communication_range)
                
                if distance <= max_range:
                    # Calculate quality based on the range that *made* the connection
                    los_quality = 1.0 - (distance / max_range) ** 2
                    
                    signal_boost = 1.2 if (drone1.drone_type == DroneType.LEADER or 
                                            drone2.drone_type == DroneType.LEADER) else 1.0
                    
                    energy_factor = min(drone1.battery_level, drone2.battery_level) / 100.0
                    
                    link_quality = los_quality * signal_boost * energy_factor
                    link_quality = max(0.0, min(1.0, link_quality))

                    if link_quality > 0.1:
                        link_metrics = LinkMetrics(
                            rssi=link_quality * 100,
                            latency=distance * 0.01,
                            packet_loss=1.0 - link_quality,
                            bandwidth=10.0 * link_quality,
                            last_updated=time.time()
                        )
                        drone1.add_neighbor(drone_id2, link_metrics)
                        drone2.add_neighbor(drone_id1, link_metrics)

    def send_ant(self, source_id: str, dest_id: str) -> Optional[Dict]:
        """Send a single ant and return result"""
        if source_id not in self.drones or dest_id not in self.drones:
            return None
            
        source_drone = self.drones[source_id]
        dest_drone = self.drones[dest_id]
        
        ant = ForwardAnt(f"ant_{self.ant_counter}", source_drone, dest_drone, AntType.EXPLORATORY)
        self.ant_counter += 1
        
        current_drone = source_drone
        max_time = 1.0
        start_time = time.time()
        
        while current_drone and ant.should_continue() and (time.time() - start_time) < max_time:
            # --- THIS LINE IS MODIFIED ---
            next_hop_id = current_drone.process_forward_ant_aco(ant, self.drones)
            # --- END MODIFICATION ---
            
            if not next_hop_id or next_hop_id not in self.drones:
                break
                
            current_drone = self.drones[next_hop_id]
        
        # Process backward ant if successful
        if ant.path_taken and ant.path_taken[-1]['drone_id'] == dest_id:
            backward_ant = BackwardAnt(ant)
            backward_ant.calculate_path_metrics()
            backward_ant.update_pheromones(self.drones)
            
            route = [hop['drone_id'] for hop in ant.path_taken]
            path_quality = ant.calculate_current_path_quality()
            
            return {
                'success': True,
                'route': route,
                'hop_count': len(route) - 1,
                'path_quality': path_quality,
                'timestamp': time.time()
            }
        
        return {'success': False, 'timestamp': time.time()}

    def run_round(self, source_id: str, dest_id: str, duration: int = 10, reset_pheromones: bool = False):
        """Run a single round of ACO simulation"""
        self.round_number += 1
        self.current_round_results = []
        
        # Validate connectivity before starting
        if not self.check_path_exists(source_id, dest_id):
            print(f"\n⚠️  WARNING: No physical path exists between {source_id} and {dest_id}!")
            print("    The drones may be too far apart or disconnected.")
            self.visualize_connectivity(source_id, dest_id)
            
            retry = input("\n    Continue anyway? (y/n, default=n): ").strip().lower()
            if retry != 'y':
                print("    Skipping this round.\n")
                return
        
        if reset_pheromones:
            print("\n🔄 Resetting all pheromones to initial state...")
            for drone in self.drones.values():
                drone.reset_pheromones()
        
        print(f"\n{'='*70}")
        print(f"ROUND {self.round_number}: ACO ROUTING SIMULATION")
        print(f"{'='*70}")
        print(f"Source: {source_id} → Destination: {dest_id}")
        print(f"Duration: {duration} seconds")
        print(f"Ants per second: {self.aco_params['ants_per_second']}")
        if reset_pheromones:
            print("Status: Fresh start (pheromones reset)")
        else:
            print("Status: Continuing from previous learning")
        print(f"{'='*70}\n")
        
        self.running = True
        start_time = time.time()
        last_ant_time = 0
        ant_interval = 1.0 / self.aco_params['ants_per_second']
        
        successful_routes = 0
        failed_routes = 0
        
        try:
            while self.running and (time.time() - start_time) < duration:
                self.simulation_time = time.time() - start_time
                
                # Send ants periodically
                if self.simulation_time >= last_ant_time + ant_interval:
                    result = self.send_ant(source_id, dest_id)
                    if result:
                        self.current_round_results.append(result)
                        if result['success']:
                            successful_routes += 1
                            print(f"[{self.simulation_time:.1f}s] ✓ Ant #{self.ant_counter-1}: "
                                  f"Found route with {result['hop_count']} hops, "
                                  f"quality: {result['path_quality']*100:.1f}%")
                        else:
                            failed_routes += 1
                    last_ant_time = self.simulation_time
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n⚠️ Round interrupted by user")
        finally:
            self.running = False
            
        # Round summary
        print(f"\n{'-'*70}")
        print(f"ROUND {self.round_number} SUMMARY:")
        print(f"{'-'*70}")
        print(f"Total ants sent: {successful_routes + failed_routes}")
        print(f"Successful routes: {successful_routes}")
        print(f"Failed routes: {failed_routes}")
        
        if successful_routes > 0:
            success_rate = (successful_routes / (successful_routes + failed_routes)) * 100
            print(f"Success rate: {success_rate:.1f}%")
            
            successful_results = [r for r in self.current_round_results if r['success']]
            avg_hops = np.mean([r['hop_count'] for r in successful_results])
            avg_quality = np.mean([r['path_quality'] for r in successful_results])
            
            print(f"Average hops: {avg_hops:.2f}")
            print(f"Average path quality: {avg_quality*100:.1f}%")
            
            # Best route
            best_route = max(successful_results, key=lambda x: x['path_quality'])
            print(f"\nBest route found:")
            print(f"  Path: {' → '.join(best_route['route'])}")
            print(f"  Hops: {best_route['hop_count']}")
            print(f"  Quality: {best_route['path_quality']*100:.1f}%")
        else:
            print(f"\n❌ NO SUCCESSFUL ROUTES FOUND!")
            print(f"   This usually means:")
            print(f"   • Source and destination are too far apart")
            print(f"   • No intermediate drones to relay the signal")
            print(f"   • Network is disconnected")
        
        print(f"{'='*70}\n")
        
        # Save round results
        self.round_history.append({
            'round': self.round_number,
            'source': source_id,
            'destination': dest_id,
            'duration': duration,
            'reset_pheromones': reset_pheromones,
            'results': self.current_round_results.copy(),
            'successful': successful_routes,
            'failed': failed_routes
        })
        
        # Visualize this round
        self.visualize_round(source_id, dest_id)
    
    def check_path_exists(self, source_id: str, dest_id: str) -> bool:
        """Check if a path exists using BFS"""
        if source_id not in self.drones or dest_id not in self.drones:
            return False
        
        if source_id == dest_id:
            return True
        
        visited = set()
        queue = deque([source_id])
        visited.add(source_id)
        
        while queue:
            current_id = queue.popleft()
            current_drone = self.drones[current_id]
            
            if dest_id in current_drone.neighbor_drones:
                return True
            
            for neighbor_id in current_drone.neighbor_drones.keys():
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append(neighbor_id)
        
        return False
    
    def visualize_connectivity(self, source_id: str, dest_id: str):
        """Show network connectivity for debugging"""
        print(f"\n🔍 CONNECTIVITY ANALYSIS:")
        print(f"   Source: {source_id}")
        
        source_drone = self.drones[source_id]
        print(f"   Position: ({source_drone.position.x:.0f}, {source_drone.position.y:.0f}, {source_drone.position.z:.0f})")
        print(f"   Range: {source_drone.communication_range:.0f}m")
        print(f"   Neighbors: {len(source_drone.neighbor_drones)}")
        if source_drone.neighbor_drones:
            for neighbor_id in list(source_drone.neighbor_drones.keys())[:5]:
                neighbor = self.drones[neighbor_id]
                distance = source_drone.position.distance_to(neighbor.position)
                print(f"      • {neighbor_id} (distance: {distance:.0f}m)")
        
        print(f"\n   Destination: {dest_id}")
        dest_drone = self.drones[dest_id]
        print(f"   Position: ({dest_drone.position.x:.0f}, {dest_drone.position.y:.0f}, {dest_drone.position.z:.0f})")
        print(f"   Range: {dest_drone.communication_range:.0f}m")
        print(f"   Neighbors: {len(dest_drone.neighbor_drones)}")
        if dest_drone.neighbor_drones:
            for neighbor_id in list(dest_drone.neighbor_drones.keys())[:5]:
                neighbor = self.drones[neighbor_id]
                distance = dest_drone.position.distance_to(neighbor.position)
                print(f"      • {neighbor_id} (distance: {distance:.0f}m)")
        
        # Direct distance
        direct_distance = source_drone.position.distance_to(dest_drone.position)
        print(f"\n   Direct distance: {direct_distance:.0f}m")
        print(f"   Max single hop: {max(source_drone.communication_range, dest_drone.communication_range):.0f}m")
        
        if direct_distance <= source_drone.communication_range:
            print(f"   ✓ Within direct range!")
        else:
            print(f"   ✗ Requires {(direct_distance / 1000):.1f} hops minimum")

    def show_network_overview(self):
        """Display network statistics"""
        print(f"\n📊 NETWORK OVERVIEW:")
        print(f"{'='*70}")
        
        total_drones = len(self.drones)
        leaders = sum(1 for d in self.drones.values() if d.drone_type == DroneType.LEADER)
        workers = total_drones - leaders
        
        print(f"Total drones: {total_drones}")
        print(f"  Leaders: {leaders}")
        print(f"  Workers: {workers}")
        
        # Connectivity stats
        neighbor_counts = [len(d.neighbor_drones) for d in self.drones.values()]
        if neighbor_counts:
            print(f"\nConnectivity:")
            print(f"  Avg neighbors per drone: {np.mean(neighbor_counts):.1f}")
            print(f"  Min neighbors: {min(neighbor_counts)}")
            print(f"  Max neighbors: {max(neighbor_counts)}")
            
            isolated = sum(1 for count in neighbor_counts if count == 0)
            if isolated > 0:
                print(f"  ⚠️  Isolated drones (0 neighbors): {isolated}")
        
        # Show isolated drones
        isolated_drones = [drone_id for drone_id, drone in self.drones.items() 
                          if len(drone.neighbor_drones) == 0]
        if isolated_drones:
            print(f"\n⚠️  WARNING: These drones are isolated:")
            for drone_id in isolated_drones:
                drone = self.drones[drone_id]
                print(f"     • {drone_id} at ({drone.position.x:.0f}, {drone.position.y:.0f})")
        
        print(f"{'='*70}\n")

    def visualize_round(self, source_id: str, dest_id: str):
        """Visualize the network and routes from current round"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Left: Network with best route
        successful_results = [r for r in self.current_round_results if r['success']]
        
        for drone_id, drone in self.drones.items():
            color = 'red' if drone.drone_type == DroneType.LEADER else 'blue'
            marker = 's' if drone.drone_type == DroneType.LEADER else 'o'
            size = 150 if drone.drone_type == DroneType.LEADER else 100
            alpha = 0.3
            
            if drone_id == source_id:
                color = 'green'
                alpha = 1.0
                size = 200
            elif drone_id == dest_id:
                color = 'orange'
                alpha = 1.0
                size = 200
            
            ax1.scatter(drone.position.x, drone.position.y, 
                       c=color, marker=marker, s=size, alpha=alpha,
                       edgecolors='black', linewidth=1.5, zorder=3)
            
            # Draw neighbor connections (faint)
            for neighbor_id in drone.neighbor_drones.keys():
                if neighbor_id in self.drones:
                    neighbor = self.drones[neighbor_id]
                    ax1.plot([drone.position.x, neighbor.position.x],
                           [drone.position.y, neighbor.position.y], 
                           'gray', alpha=0.1, linewidth=0.5, zorder=1)
            
            # Label important nodes
            if drone_id in [source_id, dest_id]:
                ax1.text(drone.position.x, drone.position.y + 80, 
                        drone_id, fontsize=10, ha='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        # Draw best route if exists
        if successful_results:
            best_route = max(successful_results, key=lambda x: x['path_quality'])
            route = best_route['route']
            
            for i in range(len(route) - 1):
                current = self.drones[route[i]]
                next_drone = self.drones[route[i + 1]]
                
                ax1.plot([current.position.x, next_drone.position.x],
                        [current.position.y, next_drone.position.y],
                        'lime', linewidth=4, alpha=0.8, zorder=2)
                
                # Arrow
                dx = next_drone.position.x - current.position.x
                dy = next_drone.position.y - current.position.y
                ax1.arrow(current.position.x, current.position.y,
                        dx * 0.8, dy * 0.8,
                        head_width=40, head_length=40, fc='lime', 
                        ec='lime', alpha=0.8, zorder=2, linewidth=2)
        
        ax1.set_title(f'Round {self.round_number}: Network & Best Route\n'
                     f'{source_id} → {dest_id}', 
                     fontsize=12, fontweight='bold')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Right: Performance metrics
        if self.current_round_results:
            times = [r['timestamp'] - self.current_round_results[0]['timestamp'] 
                    for r in self.current_round_results if r['success']]
            qualities = [r['path_quality'] * 100 for r in self.current_round_results if r['success']]
            
            if times and qualities:
                ax2.plot(times, qualities, 'b-o', linewidth=2, markersize=6, alpha=0.7)
                ax2.set_xlabel('Time (seconds)', fontsize=11)
                ax2.set_ylabel('Path Quality (%)', fontsize=11)
                ax2.set_title(f'Path Quality Over Time', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 100)
                
                # Add statistics box
                if qualities:
                    stats_text = f"Successful ants: {len(qualities)}\n"
                    stats_text += f"Avg quality: {np.mean(qualities):.1f}%\n"
                    stats_text += f"Best quality: {max(qualities):.1f}%"
                    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                            fontsize=10, verticalalignment='top',
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))
        
        plt.suptitle(f'ACO Routing Simulation - Round {self.round_number}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'aco_round_{self.round_number}_{source_id}_to_{dest_id}.png'
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        print(f"📊 Visualization saved: {filename}")
        plt.close()

    def show_pheromone_map(self, destination_id: str):
        """Visualize pheromone trails for a specific destination"""
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Draw all drones
        for drone_id, drone in self.drones.items():
            color = 'red' if drone.drone_type == DroneType.LEADER else 'blue'
            marker = 's' if drone.drone_type == DroneType.LEADER else 'o'
            size = 150 if drone.drone_type == DroneType.LEADER else 100
            alpha = 1.0 if drone_id == destination_id else 0.5
            
            ax.scatter(drone.position.x, drone.position.y, 
                      c=color, marker=marker, s=size, alpha=alpha,
                      edgecolors='black', linewidth=1.5, zorder=3)
            
            if drone_id == destination_id:
                ax.text(drone.position.x, drone.position.y + 80, 
                       f"DEST\n{drone_id}", fontsize=10, ha='center', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9))
        
        # Draw pheromone trails
        max_pheromone = 0.1
        for drone_id, drone in self.drones.items():
            if destination_id in drone.pheromone_table_aco:
                for neighbor_id, pheromone in drone.pheromone_table_aco[destination_id].items():
                    if neighbor_id in self.drones:
                        max_pheromone = max(max_pheromone, pheromone)
        
        for drone_id, drone in self.drones.items():
            if destination_id in drone.pheromone_table_aco:
                for neighbor_id, pheromone in drone.pheromone_table_aco[destination_id].items():
                    if neighbor_id in self.drones:
                        neighbor = self.drones[neighbor_id]
                        
                        # Line thickness based on pheromone
                        width = 1 + (pheromone / max_pheromone) * 8
                        alpha_val = 0.3 + (pheromone / max_pheromone) * 0.7
                        color_val = pheromone / max_pheromone
                        color = plt.cm.YlOrRd(color_val)
                        
                        ax.plot([drone.position.x, neighbor.position.x],
                               [drone.position.y, neighbor.position.y],
                               color=color, linewidth=width, alpha=alpha_val, zorder=2)
                        
                        # Add pheromone value
                        mid_x = (drone.position.x + neighbor.position.x) / 2
                        mid_y = (drone.position.y + neighbor.position.y) / 2
                        ax.text(mid_x, mid_y, f'{pheromone:.2f}', 
                               fontsize=7, ha='center',
                               bbox=dict(boxstyle='round,pad=0.2', 
                                       facecolor='white', alpha=0.8))
        
        ax.set_title(f'Pheromone Map to Destination: {destination_id}\n'
                    f'Round {self.round_number}', 
                    fontsize=13, fontweight='bold')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add colorbar legend
        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, 
                                   norm=plt.Normalize(vmin=0, vmax=max_pheromone))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Pheromone Strength', fontsize=11)
        
        plt.tight_layout()
        filename = f'pheromone_map_round_{self.round_number}_dest_{destination_id}.png'
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        print(f"🗺️  Pheromone map saved: {filename}")
        plt.show()

    def visualize_network_static(self):
        """Visualize the drone network's current state"""
        if not self.drones:
            print("❌ No drones to visualize.")
            return
        
        print("📊 Generating static network visualization...")
        plt.figure(figsize=(12, 10))
        
        for drone_id, drone in self.drones.items():
            color = 'red' if drone.drone_type == DroneType.LEADER else 'blue'
            marker = 's' if drone.drone_type == DroneType.LEADER else 'o'
            size = 150 if drone.drone_type == DroneType.LEADER else 100
            
            plt.scatter(drone.position.x, drone.position.y, c=color, 
                       marker=marker, s=size, alpha=0.7, edgecolors='black', linewidth=1.5)
            
            # Add drone ID label
            plt.text(drone.position.x, drone.position.y + 60, 
                    drone_id, fontsize=8, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black'))
            
            # Draw connections
            for neighbor_id in drone.neighbor_drones.keys():
                if neighbor_id in self.drones:
                    neighbor = self.drones[neighbor_id]
                    plt.plot([drone.position.x, neighbor.position.x],
                            [drone.position.y, neighbor.position.y], 
                            'gray', alpha=0.2, linewidth=0.5)
        
        leader_count = sum(1 for d in self.drones.values() if d.drone_type == DroneType.LEADER)
        worker_count = sum(1 for d in self.drones.values() if d.drone_type == DroneType.WORKER)
        
        title = f"Static Drone Network Visualization\n"
        title += f"Leaders: {leader_count} | Workers: {worker_count} | Total: {len(self.drones)}"
        plt.title(title, fontsize=13, fontweight='bold')
        
        plt.xlabel("X Position (m)", fontsize=11)
        plt.ylabel("Y Position (m)", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        red_patch = mpatches.Patch(color='red', label='Leader Drones')
        blue_patch = mpatches.Patch(color='blue', label='Worker Drones')
        plt.legend(handles=[red_patch, blue_patch], fontsize=10)
        
        plt.tight_layout()
        plt.show()


# ============= GRAPHICAL DRONE PLACEMENT GUI =============

class GraphicalDronePlacement:
    def __init__(self, area_size: Tuple[float, float] = (2000, 2000)):
        self.area_size = area_size
        self.controller = InteractiveACOController()
        self.placement_mode = DroneType.WORKER
        self.fig, self.ax = None, None
        self.z_height = 150.0
        self.mode_indicator = None
        self.placement_complete = False
        
    def start_placement_interface(self):
        print("\n" + "="*70)
        print("GRAPHICAL DRONE PLACEMENT - ACO SIMULATOR")
        print("="*70)
        print("CONTROLS:")
        print("  - LEFT CLICK: Place WORKER drone at cursor position")
        print("  - RIGHT CLICK: Place LEADER drone at cursor position")
        print("  - Press 'w': Switch to WORKER drone mode (left click)")
        print("  - Press 'l': Switch to LEADER drone mode (left click)")
        print("  - Press 'c': Clear all drones")
        print("  - Press 'd': Done placing - start ACO simulation")
        print("  - Press 'q': Quit without simulation")
        print("  - Press '+'/'-': Adjust altitude (Z-height)")
        print("="*70)
        print(f"\nArea size: {self.area_size[0]}m x {self.area_size[1]}m")
        print(f"Initial mode: {self.placement_mode.value.upper()} (left click)")
        print(f"Initial altitude: {self.z_height}m")
        print("\nTIP: Right-click always places LEADER drones!\n")
        
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.canvas.manager.set_window_title('ACO Simulator - Drone Placement')
        
        self.ax.set_xlim(0, self.area_size[0])
        self.ax.set_ylim(0, self.area_size[1])
        self.ax.set_xlabel('X Position (meters)', fontsize=12)
        self.ax.set_ylabel('Y Position (meters)', fontsize=12)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.set_aspect('equal')
        
        self._update_title()
        
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        self._add_legend()
        self._add_instructions()
        self._draw_mode_indicator()
        
        plt.tight_layout()
        plt.show()
        
        return self.placement_complete
    
    def _update_title(self):
        leader_count = sum(1 for d in self.controller.drones.values() 
                          if d.drone_type == DroneType.LEADER)
        worker_count = sum(1 for d in self.controller.drones.values() 
                          if d.drone_type == DroneType.WORKER)
        
        mode_str = f"*** {self.placement_mode.value.upper()} MODE (L-CLICK) ***"
        title = f"ACO Simulator - {mode_str}\n"
        title += f"Altitude: {self.z_height:.0f}m | Leaders: {leader_count} | Workers: {worker_count}"
        
        title_color = 'darkred' if self.placement_mode == DroneType.LEADER else 'darkblue'
        self.ax.set_title(title, fontsize=14, fontweight='bold', color=title_color)
    
    def _draw_mode_indicator(self):
        if self.mode_indicator:
            self.mode_indicator.set_visible(False)
        
        color = 'red' if self.placement_mode == DroneType.LEADER else 'blue'
        text = f"LEFT-CLICK: {self.placement_mode.value.upper()}\nRIGHT-CLICK: LEADER"
        
        self.mode_indicator = self.ax.text(
            0.02, 0.98, text, 
            transform=self.ax.transAxes,
            fontsize=11, fontweight='bold', color=color,
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', 
                     alpha=0.9, edgecolor=color, linewidth=2),
            zorder=10
        )
    
    def _add_legend(self):
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', label='Leader Drone (2000m range)',
                   markerfacecolor='red', markersize=12, alpha=0.7),
            Line2D([0], [0], marker='o', color='w', label='Worker Drone (1000m range)',
                   markerfacecolor='blue', markersize=10, alpha=0.7),
            Line2D([0], [0], color='gray', alpha=0.3, label='Communication Range', linewidth=2)
        ]
        
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    def _add_instructions(self):
        instructions = "L-Click=Mode | R-Click=Leader | w/l=Mode | c=Clear | d=Done | q=Quit | +/-=Alt"
        self.ax.text(0.5, -0.08, instructions, transform=self.ax.transAxes,
                    ha='center', fontsize=10, bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
    
    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        
        if event.xdata is None or event.ydata is None:
            return
        
        x, y = event.xdata, event.ydata
        
        # Check bounds
        if not (0 <= x <= self.area_size[0] and 0 <= y <= self.area_size[1]):
            return
        
        if event.button == 1:  # Left click - use current mode
            self._add_drone(x, y, self.z_height, self.placement_mode)
            print(f"✓ {self.placement_mode.value.upper()} drone placed at ({x:.0f}, {y:.0f}, {self.z_height:.0f})")
        elif event.button == 3:  # Right click - always place LEADER
            self._add_drone(x, y, self.z_height, DroneType.LEADER)
            print(f"✓ LEADER drone placed at ({x:.0f}, {y:.0f}, {self.z_height:.0f}) [right-click]")
        
        self._redraw_drones()
    
    def _add_drone(self, x: float, y: float, z: float, drone_type: DroneType):
        if drone_type == DroneType.LEADER:
            existing_count = sum(1 for d in self.controller.drones.values() 
                                if d.drone_type == DroneType.LEADER)
            drone_id = f"leader_{existing_count}"
            initial_energy = 150.0
        else:
            existing_count = sum(1 for d in self.controller.drones.values() 
                                if d.drone_type == DroneType.WORKER)
            drone_id = f"worker_{existing_count}"
            initial_energy = 100.0
        
        position = Position(x, y, z)
        new_drone = UnifiedDrone(drone_id, position, drone_type, initial_energy)
        self.controller.drones[drone_id] = new_drone
    
    def _on_key(self, event):
        if event.key == 'w':
            self.placement_mode = DroneType.WORKER
            print(f"\n>>> Switched to WORKER mode (altitude: {self.z_height}m) <<<")
        elif event.key == 'l':
            self.placement_mode = DroneType.LEADER
            print(f"\n>>> Switched to LEADER mode (altitude: {self.z_height}m) <<<")
        elif event.key == 'c':
            self.controller.drones.clear()
            print("\n>>> Cleared all drones <<<")
            self._redraw_drones()
            return
        elif event.key == 'd':
            if len(self.controller.drones) < 2:
                print("\n❌ Need at least 2 drones to start simulation!")
                return
            print("\n>>> Finalizing placement... <<<")
            plt.close(self.fig)
            print(f"✓ Total drones placed: {len(self.controller.drones)}")
            self.placement_complete = True
            return
        elif event.key == 'q':
            print("\n>>> Quitting placement <<<")
            self.placement_complete = False
            plt.close(self.fig)
            return
        elif event.key == '+' or event.key == '=':
            self.z_height += 10
            print(f"\n>>> Altitude set to {self.z_height:.0f}m <<<")
        elif event.key == '-' or event.key == '_':
            self.z_height = max(10.0, self.z_height - 10)
            print(f"\n>>> Altitude set to {self.z_height:.0f}m <<<")
        else:
            return
        
        # Update visuals for mode/altitude changes
        self._update_title()
        if self.mode_indicator:
            self.mode_indicator.set_visible(False)
        self.mode_indicator = None
        self._draw_mode_indicator()
        self.fig.canvas.draw()

    def _redraw_drones(self):
        self.ax.clear()
        
        # Redraw background
        self.ax.set_xlim(0, self.area_size[0])
        self.ax.set_ylim(0, self.area_size[1])
        self.ax.set_xlabel('X Position (meters)', fontsize=12)
        self.ax.set_ylabel('Y Position (meters)', fontsize=12)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.set_aspect('equal')
        
        # Redraw drones
        for drone_id, drone in self.controller.drones.items():
            if drone.drone_type == DroneType.LEADER:
                color = 'red'
                marker = 's'
                size = 150
            else:
                color = 'blue'
                marker = 'o'
                size = 100
            
            self.ax.scatter(drone.position.x, drone.position.y, 
                            c=color, marker=marker, s=size, alpha=0.7,
                            edgecolors='black', linewidth=1.5, zorder=3)
            
            # Draw communication range circle
            range_circle = Circle((drone.position.x, drone.position.y), 
                                  drone.communication_range, 
                                  color=color, fill=False, 
                                  linestyle='--', alpha=0.15, zorder=2)
            self.ax.add_patch(range_circle)
            
            # Add drone ID label
            self.ax.text(drone.position.x, drone.position.y + 80, 
                         drone_id, fontsize=8, ha='center',
                         fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Redraw persistent elements
        self._update_title()
        self._add_legend()
        self._add_instructions()
        self.mode_indicator = None
        self._draw_mode_indicator()
        
        self.fig.canvas.draw()


# ============= INTERACTIVE MAIN FUNCTION =============

def run_interactive_aco():
    print("\n" + "="*70)
    print("INTERACTIVE ACO ROUTING SIMULATOR")
    print("="*70)
    
    # Setup phase with graphical placement
    print("\n📋 SETUP PHASE")
    print("-" * 70)
    print("Choose drone placement method:")
    print("  [1] Graphical placement (click to place drones)")
    print("  [2] Random automatic placement")
    
    choice = input("\nEnter choice (1 or 2, default=1): ").strip() or "1"
    
    if choice == "1":
        # Graphical placement
        print("\n🖱️  Opening graphical placement interface...")
        print("    Click on the map to place drones!")
        placement_gui = GraphicalDronePlacement(area_size=(2000, 2000))
        placement_complete = placement_gui.start_placement_interface()
        
        if not placement_complete:
            print("\n❌ Placement cancelled. Exiting.")
            return
        
        controller = placement_gui.controller
        
        if len(controller.drones) < 2:
            print("\n❌ Need at least 2 drones. Exiting.")
            return
    else:
        # Automatic random placement
        controller = InteractiveACOController()
        num_drones = int(input("Enter number of drones (recommended 15-30): ").strip() or "20")
        controller.initialize_swarm(num_drones)
    
    # Initialize neighbor relationships
    print("\n🔄 Calculating neighbor relationships...")
    controller.update_neighbor_relationships()
    
    print("\n✓ Network initialized successfully!")
    print(f"  Total drones: {len(controller.drones)}")
    
    # Show network overview
    controller.show_network_overview()
    
    
    # Main simulation loop
    while True:

        # Show available drones
        drone_list = sorted(controller.drones.keys())
        print(f"\n📡 Available drones:")
        for i, drone_id in enumerate(drone_list, 1):
            drone = controller.drones[drone_id]
            dtype = "LEADER" if drone.drone_type == DroneType.LEADER else "WORKER"
            neighbors = len(drone.neighbor_drones)
            print(f"  [{i:2d}] {drone_id:12s} ({dtype}) - {neighbors} neighbors")

        print("\n" + "="*70)
        print(f"ROUND SELECTION (Current Round: {controller.round_number})")
        print("="*70)
        
        # Quick options
        print("\n📋 Quick options:")
        print("  [n] Show network overview")
        print("  [v] View network visualization")
        print("  [h] View round history")
        print("  [q] Quit simulator")
        print("\n  Or enter source drone to continue...")
        
        # Get source node
        print("\n🎯 Select SOURCE node:")
        source_input = input("  Enter drone ID, number, or command: ").strip()
        
        if source_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Exiting simulator. Goodbye!")
            break
        elif source_input.lower() == 'n':
            controller.show_network_overview()
            continue
        elif source_input.lower() == 'v':
            controller.visualize_network_static()
            continue
        elif source_input.lower() == 'h':
            print_round_history(controller)
            continue
        
        # Parse source
        if source_input.isdigit():
            idx = int(source_input) - 1
            if 0 <= idx < len(drone_list):
                source_id = drone_list[idx]
            else:
                print("❌ Invalid number. Try again.")
                continue
        elif source_input in controller.drones:
            source_id = source_input
        else:
            print("❌ Invalid drone ID. Try again.")
            continue
        
        # Get destination node
        print("\n🏁 Select DESTINATION node:")
        dest_input = input("  Enter drone ID or number from list above: ").strip()
        
        if dest_input.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Exiting simulator. Goodbye!")
            break
        
        # Parse destination
        if dest_input.isdigit():
            idx = int(dest_input) - 1
            if 0 <= idx < len(drone_list):
                dest_id = drone_list[idx]
            else:
                print("❌ Invalid number. Try again.")
                continue
        elif dest_input in controller.drones:
            dest_id = dest_input
        else:
            print("❌ Invalid drone ID. Try again.")
            continue
        
        # Validate source != destination
        if source_id == dest_id:
            print("❌ Source and destination must be different!")
            continue
        
        # Ask about pheromone reset
        print("\n🔄 Pheromone strategy:")
        print("  [1] Continue learning (keep pheromones from previous rounds)")
        print("  [2] Fresh start (reset all pheromones)")
        reset_choice = input("  Enter choice (1 or 2, default=1): ").strip() or "1"
        reset_pheromones = (reset_choice == "2")
        
        # Run the round
        controller.run_round(source_id, dest_id, duration=10, reset_pheromones=reset_pheromones)
        
        # Post-round options
        print("\n" + "="*70)
        print("POST-ROUND OPTIONS")
        print("="*70)
        print("  [1] Start new round with different nodes")
        print("  [2] View pheromone map for this destination")
        print("  [3] View round history summary")
        print("  [4] Quit simulator")
        
        choice = input("\nEnter choice (1-4, default=1): ").strip() or "1"
        
        if choice == "2":
            controller.show_pheromone_map(dest_id)
        elif choice == "3":
            print_round_history(controller)
        elif choice == "4":
            print("\n👋 Exiting simulator. Goodbye!")
            break
        # Otherwise loop continues for new round


def print_round_history(controller: InteractiveACOController):
    """Print summary of all rounds"""
    if not controller.round_history:
        print("\n📊 No rounds completed yet.")
        return
    
    print("\n" + "="*70)
    print("ROUND HISTORY SUMMARY")
    print("="*70)
    
    for round_data in controller.round_history:
        print(f"\n🔹 Round {round_data['round']}:")
        print(f"   Route: {round_data['source']} → {round_data['destination']}")
        print(f"   Duration: {round_data['duration']}s")
        print(f"   Pheromones: {'RESET' if round_data['reset_pheromones'] else 'CONTINUED'}")
        print(f"   Results: {round_data['successful']} successful, {round_data['failed']} failed")
        
        if round_data['successful'] > 0:
            success_rate = (round_data['successful'] / 
                          (round_data['successful'] + round_data['failed'])) * 100
            print(f"   Success rate: {success_rate:.1f}%")
            
            successful = [r for r in round_data['results'] if r['success']]
            avg_quality = np.mean([r['path_quality'] for r in successful])
            print(f"   Avg path quality: {avg_quality*100:.1f}%")
    
    print("\n" + "="*70)
    
    # Overall statistics
    total_successful = sum(r['successful'] for r in controller.round_history)
    total_failed = sum(r['failed'] for r in controller.round_history)
    
    if total_successful + total_failed > 0:
        overall_success = (total_successful / (total_successful + total_failed)) * 100
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Total rounds: {len(controller.round_history)}")
        print(f"   Total ants sent: {total_successful + total_failed}")
        print(f"   Overall success rate: {overall_success:.1f}%")
        print("="*70)


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════╗
║           INTERACTIVE ACO ROUTING SIMULATOR                         ║
║                 Ant Colony Optimization for Drone Networks          ║
╚════════════════════════════════════════════════════════════════════╝

This simulator allows you to:
  • Set up a drone network
  • Choose source and destination nodes for each round
  • Run 10-second ACO simulations
  • Continue learning or reset pheromones between rounds
  • Visualize results and pheromone trails
    """)
    
    input("Press ENTER to start the simulator...")
    
    try:
        run_interactive_aco()
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulator interrupted by user. Exiting...")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✓ Simulator closed.")
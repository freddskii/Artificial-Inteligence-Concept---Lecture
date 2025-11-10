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

class AlgorithmType(Enum):
    ACO = "ACO"
    GA = "GA"

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

@dataclass
class RouteResult:
    """Store results of a routing attempt"""
    algorithm: AlgorithmType
    source: str
    destination: str
    route: Optional[List[str]]
    latency: float
    hop_count: int
    success: bool
    timestamp: float
    path_quality: float = 0.0

# ============= ADVANCED ACO COMPONENTS =============

class ForwardAnt:
    def __init__(self, ant_id: str, source_drone: 'UnifiedDrone', destination_drone: 'UnifiedDrone', 
                 ant_type: AntType = AntType.EXPLORATORY, path_to_repair: Optional[List[str]] = None):
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
        
        # --- NEW ---
        # The old path we are trying to follow/repair
        self.path_to_repair = path_to_repair
        # --- END NEW ---
        
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
        if not self.forward_ant.path_taken:
            return {}
            
        total_quality = sum(hop['link_quality'] for hop in self.forward_ant.path_taken)
        avg_quality = total_quality / len(self.forward_ant.path_taken)
        hop_penalty = len(self.forward_ant.path_taken) * 0.1
        
        # Apply emergency boost if needed
        if self.emergency_flag:
            avg_quality *= 1.2
        
        self.path_quality_score = avg_quality - hop_penalty
        return {
            'quality_score': self.path_quality_score,
            'hop_count': len(self.forward_ant.path_taken),
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
                # Pass destination_id to the update function
                current_drone.update_pheromone_aco(destination_id, next_drone_id, path_quality)
                
                # Update routing table if quality is good
                if path_quality > 0.5:
                    current_drone.routing_table_aco[self.forward_ant.destination_drone.drone_id] = next_drone_id

# ============= GA COMPONENTS =============

class GARouteChromosome:
    def __init__(self, route: List[str], source: str, destination: str):
        self.route = route
        self.source = source
        self.destination = destination
        self.fitness = 0.0
        
    def calculate_fitness(self, drone_network: Dict[str, 'UnifiedDrone']) -> float:
        if not self.route or self.route[0] != self.source or self.route[-1] != self.destination:
            return 0.0
            
        total_quality = 0.0
        hop_count = len(self.route) - 1
        
        for i in range(hop_count):
            current_id = self.route[i]
            next_id = self.route[i + 1]
            
            if current_id in drone_network and next_id in drone_network:
                current = drone_network[current_id]
                if next_id in current.neighbor_drones:
                    link_metrics = current.neighbor_drones[next_id]
                    total_quality += link_metrics.rssi / 100.0
                else:
                    return 0.0
            else:
                return 0.0
        
        if hop_count == 0:
            return 0.0
            
        avg_quality = total_quality / hop_count
        hop_penalty = hop_count * 0.05
        
        self.fitness = max(0.1, avg_quality - hop_penalty)
        return self.fitness

# ============= UNIFIED DRONE CLASS WITH ADVANCED ACO =============

class UnifiedDrone:
    """Drone that supports both Advanced ACO and GA routing"""
    def __init__(self, drone_id: str, position: Position, drone_type: DroneType, 
                 initial_energy: float = 100.0):
        self.drone_id = drone_id
        self.position = position
        self.drone_type = drone_type
        self.battery_level = initial_energy
        self.initial_energy = initial_energy
        self.communication_range = 2000.0 if drone_type == DroneType.LEADER else 1000.0
        
        # Shared components
        self.neighbor_drones = {}
        
        # Advanced ACO components
        self.pheromone_table_aco = {}
        self.routing_table_aco = {}
        self.ants_processed = 0
        self.ant_queue = deque()
        
        # GA components
        self.route_cache_ga = {}
        
        # Performance tracking
        self.packets_forwarded_aco = 0
        self.packets_forwarded_ga = 0
        self.energy_used = 0.0
        
        self.lock = threading.RLock()
        
    def update_position(self, new_position: Position):
        distance_moved = self.position.distance_to(new_position)
        movement_energy = distance_moved * 0.01
        
        with self.lock:
            self.position = new_position
            self.battery_level -= movement_energy
            self.energy_used += movement_energy
            
            if distance_moved > 100.0:
                self.route_cache_ga.clear()
            
    def measure_link_quality(self, neighbor_drone: 'UnifiedDrone') -> float:
        """Advanced link quality measurement"""
        distance = self.position.distance_to(neighbor_drone.position)
        
        if distance > self.communication_range:
            return 0.0
            
        # Line-of-sight quality based on distance
        los_quality = 1.0 - (distance / self.communication_range) ** 2
        
        # Signal boost for leader drones
        if self.drone_type == DroneType.LEADER or neighbor_drone.drone_type == DroneType.LEADER:
            signal_boost = 1.2
        else:
            signal_boost = 1.0
            
        # Energy factor
        energy_factor = min(self.battery_level, neighbor_drone.battery_level) / 100.0
        
        # Combined link quality
        link_quality = los_quality * signal_boost * energy_factor
        return max(0.0, min(1.0, link_quality))
    
    def calculate_heuristic(self, neighbor_drone: 'UnifiedDrone', destination: 'UnifiedDrone') -> float:
        """Advanced heuristic calculation for routing decisions"""
        # Link quality component
        los_quality = self.measure_link_quality(neighbor_drone)
        
        # Signal strength component
        distance_to_neighbor = self.position.distance_to(neighbor_drone.position)
        signal_strength = 1.0 - (distance_to_neighbor / self.communication_range)
        
        # Battery compatibility
        battery_compatibility = min(self.battery_level, neighbor_drone.battery_level) / 100.0
        
        # Progress toward destination
        current_to_dest = self.position.distance_to(destination.position)
        neighbor_to_dest = neighbor_drone.position.distance_to(destination.position)
        progress = 1.0 if neighbor_to_dest < current_to_dest else 0.5
        
        # Weighted combination
        heuristic = (0.30 * los_quality + 
                    0.25 * signal_strength + 
                    0.20 * battery_compatibility + 
                    0.25 * progress)
        
        return max(0.1, heuristic)
    
    # ===== ADVANCED ACO METHODS =====
    
    def process_forward_ant_aco(self, ant: ForwardAnt) -> Optional[str]:
        with self.lock:
            self.ants_processed += 1
            
            # Calculate the real link quality of the hop we just made
            link_quality = 1.0  # Default for the first hop (the source drone)
            if ant.path_taken:
                previous_drone_id = ant.path_taken[-1]['drone_id']
                if previous_drone_id in self.neighbor_drones:
                    link_metrics = self.neighbor_drones[previous_drone_id]
                    link_quality = link_metrics.rssi / 100.0
                else:
                    link_quality = 0.1 
            
            ant.record_hop(self.drone_id, link_quality, self.position)
            
            if self.drone_id == ant.destination_drone.drone_id:
                return None # Reached destination
            if not ant.should_continue():
                return None # Ant expired

            # --- NEW REPAIR LOGIC ---
            if ant.path_to_repair:
                # Check if we are still on the original path
                # ant.hop_count is 1-based (source is 1st hop)
                # path_to_repair is 0-indexed (source is index 0)
                if self.drone_id == ant.path_to_repair[ant.hop_count - 1]:
                    # We are on the path. Check if the *next* step is valid.
                    if ant.hop_count < len(ant.path_to_repair):
                        old_next_hop_id = ant.path_to_repair[ant.hop_count]
                        
                        # Check if the old next hop is still a neighbor
                        if old_next_hop_id in self.neighbor_drones:
                            # Path is still valid. Follow it.
                            return old_next_hop_id
                    # else: we are at the last node (destination), let logic fall through
                
                # If we are here, the path is broken or we're off-path.
                # Stop following and start exploring.
                ant.path_to_repair = None 
            # --- END OF REPAIR LOGIC ---

            # If path_to_repair is None (or just broke), explore normally:
            available_neighbors = self.get_available_neighbors_aco(ant)
            if not available_neighbors:
                return None
                
            next_hop_id = self.advanced_routing_decision_aco(
                available_neighbors, ant.destination_drone, ant.ant_type
            )
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
                                     ant_type: AntType = AntType.EXPLORATORY) -> str:
        """Advanced probabilistic routing with heuristic integration"""
        if not available_neighbors:
            return None
            
        probabilities = []
        total = 0.0
        
        # Adaptive alpha/beta based on ant type
        if ant_type == AntType.EMERGENCY:
            alpha = 0.4  # Less reliance on pheromones
            beta = 0.6   # More on heuristic
        elif ant_type == AntType.DATA_COLLECTION:
            alpha = 0.7  # More on learned paths
            beta = 0.3
        else:  # EXPLORATORY
            alpha = 0.6
            beta = 0.4
        
        for neighbor_id in available_neighbors:
            # Initialize pheromone table structure
            if destination.drone_id not in self.pheromone_table_aco:
                self.pheromone_table_aco[destination.drone_id] = {}
            if neighbor_id not in self.pheromone_table_aco[destination.drone_id]:
                self.pheromone_table_aco[destination.drone_id][neighbor_id] = 0.1
                
            # Get pheromone level
            pheromone = self.pheromone_table_aco[destination.drone_id][neighbor_id]
            
            # === THIS IS THE FIX ===
            # Calculate a real heuristic based on available link metrics
            # instead of just 'heuristic = 0.7'
            heuristic = 0.1  # Default low value
            if neighbor_id in self.neighbor_drones:
                link_metrics = self.neighbor_drones[neighbor_id]
                
                # 1. Use RSSI as a measure of link quality
                link_quality = link_metrics.rssi / 100.0
                
                # 2. Use latency as a proxy for distance (lower is better)
                # Normalize latency. Assume 20.0 (2000m * 0.01) is max.
                # 1.0 - (latency / 20.0) gives a "closeness" score.
                distance_score = max(0, 1.0 - (link_metrics.latency / 20.0))
                
                # Combine factors (70% quality, 30% distance)
                heuristic = (0.7 * link_quality) + (0.3 * distance_score)
                heuristic = max(0.1, heuristic) # Ensure it's non-zero
            # === END OF FIX ===
            
            # Calculate probability
            probability = (pheromone ** alpha) * (heuristic ** beta)
            probabilities.append((neighbor_id, probability))
            total += probability
            
        # Roulette wheel selection
        if total > 0:
            random_value = random.uniform(0, total)
            cumulative = 0.0
            for neighbor_id, prob in probabilities:
                cumulative += prob
                if random_value <= cumulative:
                    return neighbor_id
                    
        return random.choice(available_neighbors)
    
    def update_pheromone_aco(self, destination_id: str, neighbor_id: str, path_quality: float):
        """Advanced pheromone update with evaporation"""
        evaporation_rate = 0.3
        Q = 2.0
        
        with self.lock:
            # Ensure destination entry exists
            if destination_id not in self.pheromone_table_aco:
                self.pheromone_table_aco[destination_id] = {}
            
            # Get current pheromone level for this specific destination-neighbor pair
            current_pheromone = self.pheromone_table_aco[destination_id].get(neighbor_id, 0.1)
            
            # Apply evaporation
            evaporated_pheromone = current_pheromone * (1 - evaporation_rate)
            
            # Add reinforcement
            reinforcement = Q * path_quality
            new_pheromone = evaporated_pheromone + reinforcement
            
            # Clamp to valid range
            new_pheromone = max(0.1, min(1.0, new_pheromone))
            
            # Store the updated value correctly
            self.pheromone_table_aco[destination_id][neighbor_id] = new_pheromone
    
    def route_packet_aco(self, destination_id: str) -> Optional[str]:
        """Route packet using learned ACO routing table"""
        with self.lock:
            if destination_id in self.routing_table_aco:
                self.packets_forwarded_aco += 1
                # Consume energy for transmission
                transmission_energy = 0.05
                self.battery_level -= transmission_energy
                self.energy_used += transmission_energy
                return self.routing_table_aco[destination_id]
            return None
    
    # ===== GA METHODS =====
    
    def generate_random_route_ga(self, destination_id: str, drone_network: Dict[str, 'UnifiedDrone'], 
                                 max_hops: int = 10) -> Optional[List[str]]:
        route = [self.drone_id]
        current_id = self.drone_id
        visited = {self.drone_id}
        
        for _ in range(max_hops):
            if current_id == destination_id:
                return route
                
            current_drone = drone_network.get(current_id)
            if not current_drone:
                return None
            
            available = [nid for nid in current_drone.neighbor_drones.keys() 
                        if nid not in visited]
            
            if destination_id in available:
                route.append(destination_id)
                return route
            
            if not available:
                return None
            
            next_hop = random.choice(available)
            route.append(next_hop)
            visited.add(next_hop)
            current_id = next_hop
            
        return route if route[-1] == destination_id else None
    
    def find_route_ga(self, destination_id: str, drone_network: Dict[str, 'UnifiedDrone'],
                     population_size: int = 20, generations: int = 10) -> Optional[List[str]]:
        # Generate initial population
        population = []
        max_attempts = population_size * 3
        attempts = 0
        
        while len(population) < population_size and attempts < max_attempts:
            attempts += 1
            route = self.generate_random_route_ga(destination_id, drone_network)
            if route:
                chrom = GARouteChromosome(route, self.drone_id, destination_id)
                chrom.calculate_fitness(drone_network)
                if chrom.fitness > 0:
                    population.append(chrom)
        
        if len(population) < 2:
            if destination_id in self.neighbor_drones:
                return [self.drone_id, destination_id]
            return None
        
        # Evolution
        for gen in range(generations):
            population.sort(key=lambda c: c.fitness, reverse=True)
            best = population[0]
            
            new_pop = [best]
            parent_pool_size = max(2, len(population) // 2)
            
            while len(new_pop) < population_size:
                if len(population) >= 2 and parent_pool_size >= 2:
                    try:
                        p1, p2 = random.sample(population[:parent_pool_size], 2)
                        child_route = self.crossover_ga(p1.route, p2.route, destination_id, drone_network)
                        if child_route:
                            child = GARouteChromosome(child_route, self.drone_id, destination_id)
                            child.calculate_fitness(drone_network)
                            if child.fitness > 0:
                                new_pop.append(child)
                            else:
                                new_pop.append(best)
                        else:
                            new_pop.append(best)
                    except ValueError:
                        new_pop.append(best)
                else:
                    break
            
            if len(new_pop) > 0:
                population = new_pop
            else:
                break
        
        if population:
            best = max(population, key=lambda c: c.fitness)
            return best.route if best.fitness > 0.1 else None
        
        return None
    
    def crossover_ga(self, route1: List[str], route2: List[str], destination: str,
                    drone_network: Dict[str, 'UnifiedDrone']) -> Optional[List[str]]:
        if len(route1) < 2:
            return route1.copy()
        
        cut_point = len(route1) // 2
        child = route1[:cut_point]
        
        for node in route2:
            if node not in child:
                child.append(node)
        
        if child[0] != self.drone_id:
            child.insert(0, self.drone_id)
        if child[-1] != destination:
            if destination in self.neighbor_drones:
                child.append(destination)
        
        return child
    
    def route_packet_ga(self, destination_id: str, drone_network: Dict[str, 'UnifiedDrone']) -> Optional[str]:
        if destination_id in self.neighbor_drones:
            self.packets_forwarded_ga += 1
            return destination_id
        
        route = self.find_route_ga(destination_id, drone_network)
        if route and len(route) > 1:
            self.packets_forwarded_ga += 1
            return route[1]
        
        return None
    
    def add_neighbor(self, neighbor_id: str, link_metrics: LinkMetrics):
        with self.lock:
            self.neighbor_drones[neighbor_id] = link_metrics

    def reset_pheromones(self):
        """Reset all pheromones to initial values"""
        with self.lock:
            self.pheromone_table_aco = {}
            self.routing_table_aco = {}

# ============= COMPARISON CONTROLLER =============

class ACOvsGAController:
    def __init__(self):
        self.drones: Dict[str, UnifiedDrone] = {}
        self.running = False
        self.simulation_time = 0
        self.ant_counter = 0  # <-- THIS IS THE FIX
        
        # --- NEW ---
        # Stores the best route found for a (source, dest) pair
        self.established_routes_aco: Dict[Tuple[str, str], RouteResult] = {}
        # --- END NEW ---
        
        # ACO parameters
        self.aco_params = {
            'alpha': 0.6,
            'beta': 0.4,
            'rho': 0.3,
            'Q': 2.0,
            'ant_generation_interval': 5.0
        }
        
        # Comparison results
        self.comparison_results = []
        self.test_interval = 5
        self.last_test_time = 0
        
        # Performance metrics
        self.aco_metrics = {
            'routes_found': 0,
            'routes_failed': 0,
            'total_hops': [],
            'latencies': [],
            'success_rate': [],
            'path_qualities': []
        }
        
        self.ga_metrics = {
            'routes_found': 0,
            'routes_failed': 0,
            'total_hops': [],
            'latencies': [],
            'success_rate': [],
            'path_qualities': []
        }
        
    def initialize_swarm(self, num_drones: int, area_size: Tuple[float, float, float] = (2000, 2000, 300)):
        print(f"Initializing unified swarm with {num_drones} drones...")
        
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
        
        for i, drone_id1 in enumerate(drone_ids):
            drone1 = self.drones[drone_id1]
            drone1.neighbor_drones.clear()
            
            for drone_id2 in drone_ids[i+1:]:
                drone2 = self.drones[drone_id2]
                distance = drone1.position.distance_to(drone2.position)
                
                if distance <= drone1.communication_range:
                    link_quality = drone1.measure_link_quality(drone2)
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

    def update_drone_positions(self):
        for drone in self.drones.values():
            new_position = Position(
                x=drone.position.x + random.uniform(-50, 50),
                y=drone.position.y + random.uniform(-50, 50),
                z=drone.position.z + random.uniform(-10, 10)
            )
            
            new_position.x = max(0, min(2000, new_position.x))
            new_position.y = max(0, min(2000, new_position.y))
            new_position.z = max(10, min(300, new_position.z))
            
            drone.update_position(new_position)

    # Advanced ACO routing test
    # RENAMED from test_aco_routing
    def _send_single_ant_aco(self, source_id: str, dest_id: str) -> RouteResult:
        start_time = time.time()
        source_drone = self.drones[source_id]
        dest_drone = self.drones[dest_id]
        
        # Create forward ant with advanced features
        ant = ForwardAnt(f"ant_{self.ant_counter}", source_drone, dest_drone, AntType.EXPLORATORY)
        self.ant_counter += 1
        
        current_drone = source_drone
        max_time = 2.0
        
        while current_drone and ant.should_continue() and (time.time() - start_time) < max_time:
            next_hop_id = current_drone.process_forward_ant_aco(ant)
            
            if not next_hop_id or next_hop_id not in self.drones:
                break
                
            current_drone = self.drones[next_hop_id]
        
        # Process backward ant if successful
        if ant.path_taken and ant.path_taken[-1]['drone_id'] == dest_id:
            backward_ant = BackwardAnt(ant)
            backward_ant.calculate_path_metrics()
            backward_ant.update_pheromones(self.drones)
            
            route = [hop['drone_id'] for hop in ant.path_taken]
            latency = time.time() - start_time
            path_quality = ant.calculate_current_path_quality()
            
            return RouteResult(
                algorithm=AlgorithmType.ACO,
                source=source_id,
                destination=dest_id,
                route=route,
                latency=latency,
                hop_count=len(route) - 1,
                success=True,
                timestamp=time.time(),
                path_quality=path_quality
            )
        else:
            return RouteResult(
                algorithm=AlgorithmType.ACO,
                source=source_id,
                destination=dest_id,
                route=None,
                latency=time.time() - start_time,
                hop_count=0,
                success=False,
                timestamp=time.time(),
                path_quality=0.0
            )
        
    def _send_single_ant_aco(self, source_id: str, dest_id: str) -> RouteResult:
        start_time = time.time()
        source_drone = self.drones[source_id]
        dest_drone = self.drones[dest_id]
        
        # Create forward ant with advanced features
        ant = ForwardAnt(f"ant_{self.ant_counter}", source_drone, dest_drone, AntType.EXPLORATORY)
        self.ant_counter += 1
        
        current_drone = source_drone
        max_time = 2.0
        
        while current_drone and ant.should_continue() and (time.time() - start_time) < max_time:
            next_hop_id = current_drone.process_forward_ant_aco(ant)
            
            if not next_hop_id or next_hop_id not in self.drones:
                break
                
            current_drone = self.drones[next_hop_id]
        
        # Process backward ant if successful
        if ant.path_taken and ant.path_taken[-1]['drone_id'] == dest_id:
            backward_ant = BackwardAnt(ant)
            backward_ant.calculate_path_metrics()
            backward_ant.update_pheromones(self.drones)
            
            route = [hop['drone_id'] for hop in ant.path_taken]
            latency = time.time() - start_time
            path_quality = ant.calculate_current_path_quality()
            
            return RouteResult(
                algorithm=AlgorithmType.ACO,
                source=source_id,
                destination=dest_id,
                route=route,
                latency=latency,
                hop_count=len(route) - 1,
                success=True,
                timestamp=time.time(),
                path_quality=path_quality
            )
        else:
            return RouteResult(
                algorithm=AlgorithmType.ACO,
                source=source_id,
                destination=dest_id,
                route=None,
                latency=time.time() - start_time,
                hop_count=0,
                success=False,
                timestamp=time.time(),
                path_quality=0.0
            )
        
    def test_aco_routing(self, source_id: str, dest_id: str) -> RouteResult:
        """
        Runs a full ACO search for a specific route over a set duration.
        This is the "head-to-head" competitor to test_ga_routing.
        """
        start_time = time.time()
        
        # --- THIS IS THE CRITICAL FIX ---
        # Reset all pheromones to give ACO a fresh start,
        # making it a fair comparison against the stateless GA.
        self.reset_all_pheromones()
        # --- END OF FIX ---

        # --- Parameters for the test ---
        test_duration = 2.0  # Give ACO 2 seconds to find the best route
        ants_per_second = 25 # Send 25 ants per second
        ant_interval = 1.0 / ants_per_second
        # ---------------------------------
        
        successful_routes: List[RouteResult] = []
        last_ant_time = 0.0
        
        while (time.time() - start_time) < test_duration:
            current_test_time = time.time() - start_time
            
            # Send ants periodically
            if current_test_time >= last_ant_time + ant_interval:
                result = self._send_single_ant_aco(source_id, dest_id)
                if result.success:
                    successful_routes.append(result)
                last_ant_time = current_test_time
            
            # Don't sleep the full 0.1, just enough to not block
            time.sleep(0.01) 
        
        # After 2 seconds, find the best route found
        if successful_routes:
            # Find the route with the highest path quality
            best_route = max(successful_routes, key=lambda r: r.path_quality)
            
            # Set the latency to the *total* time the search took
            best_route.latency = time.time() - start_time
            
            # --- NEW ---
            # Save this as the new "best" route for maintenance
            self.established_routes_aco[(source_id, dest_id)] = best_route
            # --- END NEW ---
            
            return best_route
        else:
            # No route was found in the 2-second window
            return RouteResult(
                algorithm=AlgorithmType.ACO,
                source=source_id,
                destination=dest_id,
                route=None,
                latency=time.time() - start_time,
                hop_count=0,
                success=False,
                timestamp=time.time(),
                path_quality=0.0
            )

    # GA routing test (unchanged)
    def test_ga_routing(self, source_id: str, dest_id: str) -> RouteResult:
        start_time = time.time()
        source_drone = self.drones[source_id]
        
        route = source_drone.find_route_ga(dest_id, self.drones, population_size=20, generations=10)
        latency = time.time() - start_time
        
        # Calculate path quality for GA
        path_quality = 0.0
        if route and len(route) > 1:
            total_quality = 0.0
            for i in range(len(route) - 1):
                current = self.drones[route[i]]
                next_drone_id = route[i + 1]
                if next_drone_id in current.neighbor_drones:
                    link_metrics = current.neighbor_drones[next_drone_id]
                    total_quality += link_metrics.rssi / 100.0
            if len(route) > 1:
                path_quality = total_quality / (len(route) - 1)
        
        if route and route[-1] == dest_id:
            return RouteResult(
                algorithm=AlgorithmType.GA,
                source=source_id,
                destination=dest_id,
                route=route,
                latency=latency,
                hop_count=len(route) - 1,
                success=True,
                timestamp=time.time(),
                path_quality=path_quality
            )
        else:
            return RouteResult(
                algorithm=AlgorithmType.GA,
                source=source_id,
                destination=dest_id,
                route=None,
                latency=latency,
                hop_count=0,
                success=False,
                timestamp=time.time(),
                path_quality=0.0
            )

    def visualize_test_round(self, test_pairs, aco_results, ga_results, test_number):
        """Visualize the routes found in this test round"""
        num_tests = len(test_pairs)
        fig = plt.figure(figsize=(16, 4 * num_tests))
        
        for idx, (source, dest) in enumerate(test_pairs):
            aco_result = aco_results[idx]
            ga_result = ga_results[idx]
            
            # ACO visualization (left column)
            ax_aco = plt.subplot(num_tests, 2, idx * 2 + 1)
            self._draw_network_with_route(ax_aco, aco_result, "Advanced ACO", source, dest)
            
            # GA visualization (right column)
            ax_ga = plt.subplot(num_tests, 2, idx * 2 + 2)
            self._draw_network_with_route(ax_ga, ga_result, "GA", source, dest)
        
        plt.suptitle(f'Test Round at {self.simulation_time}s - Advanced ACO vs GA Comparison', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        filename = f'test_round_{test_number}_time_{self.simulation_time}s.png'
        plt.savefig(filename, dpi=120, bbox_inches='tight')
        print(f"\n  → Visualization saved: {filename}")
        plt.close()
    
    def _draw_network_with_route(self, ax, result: RouteResult, algo_name: str, source: str, dest: str):
        """Draw network with highlighted route and quality metrics"""
        # Draw all drones
        for drone_id, drone in self.drones.items():
            if drone.drone_type == DroneType.LEADER:
                color = 'red'
                marker = 's'
                size = 100
                alpha = 0.3
            else:
                color = 'blue'
                marker = 'o'
                size = 70
                alpha = 0.3
            
            # Highlight source and destination
            if drone_id == source:
                color = 'green'
                alpha = 1.0
                size = 150
            elif drone_id == dest:
                color = 'orange'
                alpha = 1.0
                size = 150
            elif result.route and drone_id in result.route:
                alpha = 0.8
            
            ax.scatter(drone.position.x, drone.position.y, 
                      c=color, marker=marker, s=size, alpha=alpha,
                      edgecolors='black', linewidth=1.2, zorder=3)
            
            # Add labels for important nodes
            if drone_id in [source, dest] or (result.route and drone_id in result.route):
                fontsize = 8 if drone_id in [source, dest] else 7
                fontweight = 'bold' if drone_id in [source, dest] else 'normal'
                ax.text(drone.position.x, drone.position.y + 60, 
                       drone_id, fontsize=fontsize, ha='center',
                       fontweight=fontweight,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        # Draw neighbor connections (faint)
        for drone_id, drone in self.drones.items():
            for neighbor_id in drone.neighbor_drones.keys():
                if neighbor_id in self.drones:
                    neighbor = self.drones[neighbor_id]
                    ax.plot([drone.position.x, neighbor.position.x],
                           [drone.position.y, neighbor.position.y], 
                           'gray', alpha=0.1, linewidth=0.5, zorder=1)
        
        # Calculate route quality if successful
        route_quality = result.path_quality
        total_distance = 0.0
        link_qualities = []
        
        if result.success and result.route:
            for i in range(len(result.route) - 1):
                current = self.drones[result.route[i]]
                next_drone = self.drones[result.route[i + 1]]
                
                # Calculate link quality
                if result.route[i + 1] in current.neighbor_drones:
                    link_metrics = current.neighbor_drones[result.route[i + 1]]
                    link_quality = link_metrics.rssi / 100.0
                    link_qualities.append(link_quality)
                
                # Calculate distance
                distance = current.position.distance_to(next_drone.position)
                total_distance += distance
                
                # Draw route line with color based on quality
                color_intensity = link_quality if link_qualities else 0.7
                line_color = plt.cm.RdYlGn(color_intensity)
                
                ax.plot([current.position.x, next_drone.position.x],
                       [current.position.y, next_drone.position.y],
                       color=line_color, linewidth=4, alpha=0.8, zorder=2)
                
                # Add arrow
                dx = next_drone.position.x - current.position.x
                dy = next_drone.position.y - current.position.y
                ax.arrow(current.position.x, current.position.y,
                        dx * 0.8, dy * 0.8,
                        head_width=30, head_length=30, fc=line_color, 
                        ec=line_color, alpha=0.8, zorder=2, linewidth=1.5)
                
                # Add hop number with link quality
                mid_x = (current.position.x + next_drone.position.x) / 2
                mid_y = (current.position.y + next_drone.position.y) / 2
                hop_text = f"#{i+1}\n{link_quality*100:.0f}%" if link_qualities else f"#{i+1}"
                ax.text(mid_x, mid_y, hop_text, fontsize=7,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9),
                       ha='center', fontweight='bold', zorder=4)
        
        # Title with detailed metrics
        if result.success:
            title = f"{algo_name}: ✓ SUCCESS\n"
            title += f"{source} → {dest}\n"
            title += f"Hops: {result.hop_count} | Distance: {total_distance:.0f}m\n"
            title += f"Quality: {route_quality*100:.1f}% | Latency: {result.latency*1000:.1f}ms"
            
            # Color based on quality
            if route_quality >= 0.7:
                title_color = 'darkgreen'
            elif route_quality >= 0.4:
                title_color = 'darkorange'
            else:
                title_color = 'darkred'
        else:
            title = f"{algo_name}: ✗ FAILED\n"
            title += f"{source} → {dest}\n"
            title += f"No route found\n"
            title += f"Latency: {result.latency*1000:.1f}ms"
            title_color = 'darkred'
        
        ax.set_title(title, fontsize=9, fontweight='bold', color=title_color, pad=8)
        ax.set_xlabel('X Position (m)', fontsize=8)
        ax.set_ylabel('Y Position (m)', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add quality legend if route exists
        if result.success and link_qualities:
            legend_text = f"Avg Link Quality: {route_quality*100:.1f}%\n"
            legend_text += f"Min Quality: {min(link_qualities)*100:.1f}%\n"
            legend_text += f"Max Quality: {max(link_qualities)*100:.1f}%"
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes,
                   fontsize=7, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))
    
    def run_comparison_test(self):
        """Run comparison test between Advanced ACO and GA"""
        drone_ids = list(self.drones.keys())
        if len(drone_ids) < 2:
            return
        
        # Select 5 random source-destination pairs
        test_pairs = []
        for _ in range(min(5, len(drone_ids) // 2)):
            source = random.choice(drone_ids)
            dest = random.choice([d for d in drone_ids if d != source])
            test_pairs.append((source, dest))
        
        print(f"\n{'='*70}")
        print(f"COMPARISON TEST AT TIME {self.simulation_time}s")
        print(f"{'='*70}")
        
        aco_results = []
        ga_results = []
        
        for source, dest in test_pairs:
            print(f"\nTesting route: {source} → {dest}")
            
            # Test Advanced ACO
            aco_result = self.test_aco_routing(source, dest)
            aco_results.append(aco_result)
            
            if aco_result.success:
                print(f"  ACO: SUCCESS | Hops: {aco_result.hop_count} | "
                      f"Quality: {aco_result.path_quality*100:.1f}% | "
                      f"Latency: {aco_result.latency:.3f}s")
                print(f"       Route: {' → '.join(aco_result.route)}")
                self.aco_metrics['routes_found'] += 1
                self.aco_metrics['total_hops'].append(aco_result.hop_count)
                self.aco_metrics['latencies'].append(aco_result.latency)
                self.aco_metrics['path_qualities'].append(aco_result.path_quality)
            else:
                print(f"  ACO: FAILED | Latency: {aco_result.latency:.3f}s")
                self.aco_metrics['routes_failed'] += 1
            
            # Test GA
            ga_result = self.test_ga_routing(source, dest)
            ga_results.append(ga_result)
            
            if ga_result.success:
                print(f"  GA:  SUCCESS | Hops: {ga_result.hop_count} | "
                      f"Quality: {ga_result.path_quality*100:.1f}% | "
                      f"Latency: {ga_result.latency:.3f}s")
                print(f"       Route: {' → '.join(ga_result.route)}")
                self.ga_metrics['routes_found'] += 1
                self.ga_metrics['total_hops'].append(ga_result.hop_count)
                self.ga_metrics['latencies'].append(ga_result.latency)
                self.ga_metrics['path_qualities'].append(ga_result.path_quality)
            else:
                print(f"  GA:  FAILED | Latency: {ga_result.latency:.3f}s")
                self.ga_metrics['routes_failed'] += 1
        
        # Calculate success rates
        aco_success_rate = sum(1 for r in aco_results if r.success) / len(aco_results) * 100
        ga_success_rate = sum(1 for r in ga_results if r.success) / len(ga_results) * 100
        
        self.aco_metrics['success_rate'].append(aco_success_rate)
        self.ga_metrics['success_rate'].append(ga_success_rate)
        
        print(f"\n{'-'*70}")
        print(f"Test Summary:")
        print(f"  ACO Success Rate: {aco_success_rate:.1f}%")
        print(f"  GA  Success Rate: {ga_success_rate:.1f}%")
        print(f"{'='*70}")
        
        # Generate visualization for this test round
        test_number = len(self.comparison_results) + 1
        self.visualize_test_round(test_pairs, aco_results, ga_results, test_number)
        
        self.comparison_results.append({
            'time': self.simulation_time,
            'aco_results': aco_results,
            'ga_results': ga_results
        })

    def run_simulation(self, duration: int = 60):
        print("\nStarting Advanced ACO vs GA Comparison Simulation...")
        print(f"Duration: {duration} seconds")
        print(f"Tests will run every {self.test_interval} seconds\n")
        
        self.running = True
        start_time = time.time()
        self.last_test_time = 0
        
        try:
            while self.running and (time.time() - start_time) < duration:
                self.simulation_time = int(time.time() - start_time)
                
                # Run comparison test every test_interval seconds
                if self.simulation_time >= self.last_test_time + self.test_interval:
                    self.run_comparison_test()
                    self.last_test_time = self.simulation_time
                
                # --- NEW ---
                # Run the ACO route maintenance cycle
                self.run_aco_route_maintenance()
                # --- END NEW ---
                
                # Update network
                self.update_drone_positions()
                self.update_neighbor_relationships()
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        finally:
            self.running = False
            self.print_final_comparison()
            self.visualize_comparison()

    def print_final_comparison(self):
        print("\n" + "="*70)
        print("FINAL COMPARISON RESULTS")
        print("="*70)
        
        total_tests = self.aco_metrics['routes_found'] + self.aco_metrics['routes_failed']
        
        print(f"\nAdvanced ACO Performance:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Successful Routes: {self.aco_metrics['routes_found']}")
        print(f"  Failed Routes: {self.aco_metrics['routes_failed']}")
        if self.aco_metrics['total_hops']:
            print(f"  Average Hops: {np.mean(self.aco_metrics['total_hops']):.2f}")
        if self.aco_metrics['latencies']:
            print(f"  Average Latency: {np.mean(self.aco_metrics['latencies']):.3f}s")
        if self.aco_metrics['path_qualities']:
            print(f"  Average Path Quality: {np.mean(self.aco_metrics['path_qualities'])*100:.1f}%")
        if self.aco_metrics['success_rate']:
            print(f"  Overall Success Rate: {np.mean(self.aco_metrics['success_rate']):.1f}%")
        
        print(f"\nGA Performance:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Successful Routes: {self.ga_metrics['routes_found']}")
        print(f"  Failed Routes: {self.ga_metrics['routes_failed']}")
        if self.ga_metrics['total_hops']:
            print(f"  Average Hops: {np.mean(self.ga_metrics['total_hops']):.2f}")
        if self.ga_metrics['latencies']:
            print(f"  Average Latency: {np.mean(self.ga_metrics['latencies']):.3f}s")
        if self.ga_metrics['path_qualities']:
            print(f"  Average Path Quality: {np.mean(self.ga_metrics['path_qualities'])*100:.1f}%")
        if self.ga_metrics['success_rate']:
            print(f"  Overall Success Rate: {np.mean(self.ga_metrics['success_rate']):.1f}%")

    def visualize_comparison(self):
        """Visualize comparison results"""
        if not self.comparison_results:
            print("No comparison data to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Success rate over time
        ax1 = axes[0, 0]
        if self.aco_metrics['success_rate'] and self.ga_metrics['success_rate']:
            test_times = [r['time'] for r in self.comparison_results]
            ax1.plot(test_times, self.aco_metrics['success_rate'], 'b-o', label='Advanced ACO', linewidth=2, markersize=8)
            ax1.plot(test_times, self.ga_metrics['success_rate'], 'r-s', label='GA', linewidth=2, markersize=8)
            ax1.set_xlabel('Simulation Time (s)', fontsize=11)
            ax1.set_ylabel('Success Rate (%)', fontsize=11)
            ax1.set_title('Success Rate Over Time', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
        
        # Average hop count comparison
        ax2 = axes[0, 1]
        if self.aco_metrics['total_hops'] and self.ga_metrics['total_hops']:
            data = [self.aco_metrics['total_hops'], self.ga_metrics['total_hops']]
            bp = ax2.boxplot(data, tick_labels=['Advanced ACO', 'GA'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            ax2.set_ylabel('Number of Hops', fontsize=11)
            ax2.set_title('Hop Count Distribution', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # Path quality comparison
        ax3 = axes[1, 0]
        if self.aco_metrics['path_qualities'] and self.ga_metrics['path_qualities']:
            data = [self.aco_metrics['path_qualities'], self.ga_metrics['path_qualities']]
            bp = ax3.boxplot(data, tick_labels=['Advanced ACO', 'GA'], patch_artist=True)
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            ax3.set_ylabel('Path Quality', fontsize=11)
            ax3.set_title('Path Quality Distribution', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Overall statistics bar chart
        ax4 = axes[1, 1]
        categories = ['Success\nRate (%)', 'Avg Hops', 'Avg Quality\n(%)', 'Avg Latency\n(ms)']
        
        aco_stats = [
            np.mean(self.aco_metrics['success_rate']) if self.aco_metrics['success_rate'] else 0,
            np.mean(self.aco_metrics['total_hops']) if self.aco_metrics['total_hops'] else 0,
            np.mean(self.aco_metrics['path_qualities']) * 100 if self.aco_metrics['path_qualities'] else 0,
            np.mean(self.aco_metrics['latencies']) * 1000 if self.aco_metrics['latencies'] else 0
        ]
        
        ga_stats = [
            np.mean(self.ga_metrics['success_rate']) if self.ga_metrics['success_rate'] else 0,
            np.mean(self.ga_metrics['total_hops']) if self.ga_metrics['total_hops'] else 0,
            np.mean(self.ga_metrics['path_qualities']) * 100 if self.ga_metrics['path_qualities'] else 0,
            np.mean(self.ga_metrics['latencies']) * 1000 if self.ga_metrics['latencies'] else 0
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, aco_stats, width, label='Advanced ACO', color='lightblue', edgecolor='black')
        bars2 = ax4.bar(x + width/2, ga_stats, width, label='GA', color='lightcoral', edgecolor='black')
        
        ax4.set_ylabel('Value', fontsize=11)
        ax4.set_title('Overall Performance Comparison', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories, fontsize=9)
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Advanced ACO vs GA Routing Algorithm Comparison', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig('advanced_aco_vs_ga_comparison.png', dpi=150, bbox_inches='tight')
        print("\nComparison visualization saved to: advanced_aco_vs_ga_comparison.png")
        plt.show()
    
    def visualize_network(self):
        """Visualize the drone network"""
        if not self.drones:
            return
        
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
        
        title = f"Unified Drone Network for Advanced ACO vs GA Comparison\n"
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

    def run_aco_learning_cycle(self, num_ants: int = 5):
        """Send exploratory ants to learn in the background"""
        if len(self.drones) < 2:
            return
        
        drone_ids = list(self.drones.keys())
        
        for _ in range(num_ants):
            # Pick random source/dest for exploration
            source_id = random.choice(drone_ids)
            dest_id = random.choice([d for d in drone_ids if d != source_id])
            
            source_drone = self.drones.get(source_id)
            dest_drone = self.drones.get(dest_id)
            
            if not source_drone or not dest_drone:
                continue
            
            # Send one ant. We don't care about its result,
            # only that it updates pheromones via its BackwardAnt.
            ant = ForwardAnt(f"ant_learn_{self.ant_counter}", 
                            source_drone, dest_drone, AntType.EXPLORATORY)
            self.ant_counter += 1
            
            current_drone = source_drone
            max_time = 1.0 # Give it 1 sec to find the path
            start_time = time.time()
            
            while current_drone and ant.should_continue() and (time.time() - start_time) < max_time:
                next_hop_id = current_drone.process_forward_ant_aco(ant)
                if not next_hop_id or next_hop_id not in self.drones:
                    break
                current_drone = self.drones[next_hop_id]
            
            # If the ant was successful, its backward ant will update pheromones
            if ant.path_taken and ant.path_taken[-1]['drone_id'] == dest_id:
                backward_ant = BackwardAnt(ant)
                backward_ant.calculate_path_metrics()
                backward_ant.update_pheromones(self.drones)

    def run_aco_route_maintenance(self):
        """
        Periodically checks established ACO routes.
        If a route is broken, launches a repair ant.
        """
        if not self.established_routes_aco:
            return

        # Check one established route per cycle to avoid overload
        (source_id, dest_id) = random.choice(list(self.established_routes_aco.keys()))
        route_result = self.established_routes_aco[(source_id, dest_id)]
        path = route_result.route
        
        # Check path validity
        is_valid = self.is_path_valid(path)
        
        if not is_valid:
            print(f"⚠️ ACO Route {source_id} -> {dest_id} is BROKEN! Launching repair ant.")
            # Path is broken, launch a repair ant
            self._send_repair_ant(source_id, dest_id, path)
            # Remove the old, broken path from memory
            del self.established_routes_aco[(source_id, dest_id)]

    def is_path_valid(self, path: List[str]) -> bool:
        """Checks if a given path is still valid"""
        for i in range(len(path) - 1):
            current_node_id = path[i]
            next_node_id = path[i+1]
            if current_node_id not in self.drones or next_node_id not in self.drones:
                return False
            current_drone = self.drones[current_node_id]
            if next_node_id not in current_drone.neighbor_drones:
                return False # Link is broken
        return True

    def _send_repair_ant(self, source_id: str, dest_id: str, old_path: List[str]):
        """Launches a single ant in 'repair mode'"""
        source_drone = self.drones.get(source_id)
        dest_drone = self.drones.get(dest_id)
        if not source_drone or not dest_drone:
            return

        # Create an ant with the path_to_repair
        ant = ForwardAnt(f"ant_repair_{self.ant_counter}", 
                        source_drone, dest_drone, 
                        AntType.EMERGENCY, # Use emergency for faster heuristic
                        path_to_repair=old_path)
        self.ant_counter += 1
        
        # Launch the ant (similar to _send_single_ant_aco)
        current_drone = source_drone
        max_time = 2.0
        start_time = time.time()
        
        while current_drone and ant.should_continue() and (time.time() - start_time) < max_time:
            next_hop_id = current_drone.process_forward_ant_aco(ant)
            if not next_hop_id or next_hop_id not in self.drones:
                break
            current_drone = self.drones[next_hop_id]
        
        # If repair was successful, update pheromones
        if ant.path_taken and ant.path_taken[-1]['drone_id'] == dest_id:
            print(f"✓ Repair ant found new path: {' -> '.join([hop['drone_id'] for hop in ant.path_taken])}")
            backward_ant = BackwardAnt(ant)
            backward_ant.calculate_path_metrics()
            backward_ant.update_pheromones(self.drones)
        # No else needed, if it fails, it just fails.

    def reset_all_pheromones(self):
        """Calls reset_pheromones on every drone in the network."""
        for drone in self.drones.values():
            drone.reset_pheromones()


# ============= IMPROVED INTERACTIVE PLACEMENT GUI =============

class InteractivePlacementGUI:
    def __init__(self, area_size: Tuple[float, float] = (2000, 2000)):
        self.area_size = area_size
        self.controller = ACOvsGAController()
        self.placement_mode = DroneType.WORKER
        self.fig, self.ax = None, None
        self.z_height = 150.0
        self.mode_indicator = None
        
    def start_placement_interface(self):
        print("\n" + "="*70)
        print("INTERACTIVE DRONE PLACEMENT FOR ADVANCED ACO vs GA COMPARISON")
        print("="*70)
        print("CONTROLS:")
        print("  - LEFT CLICK: Place WORKER drone at cursor position")
        print("  - RIGHT CLICK: Place LEADER drone at cursor position")
        print("  - Press 'w': Switch to WORKER drone mode (left click)")
        print("  - Press 'l': Switch to LEADER drone mode (left click)")
        print("  - Press 'c': Clear all drones")
        print("  - Press 'd': Done placing - start comparison simulation")
        print("  - Press 'q': Quit without simulation")
        print("  - Press '+'/'-': Adjust altitude (Z-height)")
        print("="*70)
        print(f"\nArea size: {self.area_size[0]}m x {self.area_size[1]}m")
        print(f"Initial mode: {self.placement_mode.value.upper()} (left click)")
        print(f"Initial altitude: {self.z_height}m")
        print("\nTIP: Right-click places LEADER drones regardless of mode!\n")
        
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.canvas.manager.set_window_title('Advanced ACO vs GA - Drone Placement')
        
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
    
    def _update_title(self):
        leader_count = sum(1 for d in self.controller.drones.values() if d.drone_type == DroneType.LEADER)
        worker_count = sum(1 for d in self.controller.drones.values() if d.drone_type == DroneType.WORKER)
        
        mode_str = f"*** {self.placement_mode.value.upper()} MODE (L-CLICK) ***"
        title = f"Advanced ACO vs GA - {mode_str}\n"
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
            transform=self.ax.transAxes, # <-- Corrected this line
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
            print(f"✓ {self.placement_mode.value.upper()} drone placed at ({x:.0f}, {y:.0f})")
        elif event.button == 3:  # Right click - always place LEADER
            self._add_drone(x, y, self.z_height, DroneType.LEADER)
            print(f"✓ LEADER drone placed at ({x:.0f}, {y:.0f}) [right-click]")
        
        self._redraw_drones()
    
    def _add_drone(self, x: float, y: float, z: float, drone_type: DroneType):
        if drone_type == DroneType.LEADER:
            existing_count = sum(1 for d in self.controller.drones.values() if d.drone_type == DroneType.LEADER)
            drone_id = f"leader_{existing_count}"
            initial_energy = 150.0
        else:
            existing_count = sum(1 for d in self.controller.drones.values() if d.drone_type == DroneType.WORKER)
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
            return # _redraw_drones handles title/indicator updates
        elif event.key == 'd':
            print("\n>>> Finalizing placement... <<<")
            plt.close(self.fig)
            print(f"Total drones placed: {len(self.controller.drones)}")
            if len(self.controller.drones) > 1:
                print("Updating initial neighbor relationships...")
                self.controller.update_neighbor_relationships()
                # Start the simulation (as intended by the help text)
                self.controller.run_simulation(duration=60)
            else:
                print("Not enough drones to run simulation. Exiting.")
            return
        elif event.key == 'q':
            print("\n>>> Quitting placement <<<")
            plt.close(self.fig)
            return
        elif event.key == '+':
            self.z_height += 10
            print(f"\n>>> Altitude set to {self.z_height:.0f}m <<<")
        elif event.key == '-':
            self.z_height = max(10.0, self.z_height - 10)
            print(f"\n>>> Altitude set to {self.z_height:.0f}m <<<")
        else:
            return # Ignore other keys
        
        # Update visuals for mode/altitude changes
        self._update_title()
        if self.mode_indicator:
            self.mode_indicator.set_visible(False)
        self.mode_indicator = None
        self._draw_mode_indicator()
        self.fig.canvas.draw()

    def _redraw_drones(self):
        self.ax.clear() # Clear the axes
        
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
                size = 100
            else:
                color = 'blue'
                marker = 'o'
                size = 70
            
            self.ax.scatter(drone.position.x, drone.position.y, 
                            c=color, marker=marker, s=size, alpha=0.7,
                            edgecolors='black', linewidth=1.2, zorder=3)
            
            # Draw communication range circle
            range_circle = Circle((drone.position.x, drone.position.y), 
                                  drone.communication_range, 
                                  color=color, fill=False, 
                                  linestyle='--', alpha=0.2, zorder=2)
            self.ax.add_patch(range_circle)
            
            # Add drone ID label
            self.ax.text(drone.position.x, drone.position.y + 60, 
                         drone_id, fontsize=8, ha='center',
                         fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        # Redraw persistent elements
        self._update_title()
        self._add_legend()
        self._add_instructions()
        self.mode_indicator = None # Force redraw
        self._draw_mode_indicator()
        
        self.fig.canvas.draw()

if __name__ == "__main__":
    # This is the main entry point of your application
    
    print("Application starting...")
    
    # 1. Create an instance of the Interactive Placement GUI
    # You can customize the area size here if you want
    gui = InteractivePlacementGUI(area_size=(2000, 2000))
    
    # 2. Start the GUI
    # This will open the matplotlib window
    gui.start_placement_interface()
    
    print("Application finished.")
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
from mpl_toolkits.mplot3d import Axes3D

RNG = 4136314
random.seed(RNG)
np.random.seed(RNG)

class DroneType(Enum):
    LEADER = "leader"
    WORKER = "worker"
    RELAY = "relay"

class MissionType(Enum):
    SURVEILLANCE = "surveillance"
    DELIVERY = "delivery"
    SEARCH_RESCUE = "search_rescue"

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

class GARouteChromosome:
    """Represents a potential route as a chromosome in GA"""
    def __init__(self, route: List[str], source: str, destination: str):
        self.route = route
        self.source = source
        self.destination = destination
        self.fitness = 0.0
        self.quality_score = 0.0
        
    def calculate_fitness(self, drone_network: Dict[str, 'GADrone']) -> float:
        """Calculate fitness based on route quality metrics"""
        if not self.route or self.route[0] != self.source or self.route[-1] != self.destination:
            return 0.0
            
        total_quality = 0.0
        total_latency = 0.0
        hop_count = len(self.route) - 1
        
        for i in range(hop_count):
            current_drone = self.route[i]
            next_drone = self.route[i + 1]
            
            if current_drone in drone_network and next_drone in drone_network:
                current = drone_network[current_drone]
                if next_drone in current.neighbor_drones:
                    link_metrics = current.neighbor_drones[next_drone]
                    total_quality += link_metrics.rssi / 100.0
                    total_latency += link_metrics.latency
                else:
                    return 0.0
            else:
                return 0.0
        
        if hop_count == 0:
            return 0.0
            
        avg_quality = total_quality / hop_count
        avg_latency = total_latency / hop_count
        
        hop_penalty = hop_count * 0.05
        latency_penalty = avg_latency * 0.1
        
        self.quality_score = avg_quality
        self.fitness = max(0.1, avg_quality - hop_penalty - latency_penalty)
        return self.fitness

class GADrone:
    def __init__(self, drone_id: str, position: Position, drone_type: DroneType, 
                 initial_energy: float = 100.0):
        self.drone_id = drone_id
        self.position = position
        self.drone_type = drone_type
        self.battery_level = initial_energy
        self.initial_energy = initial_energy
        self.communication_range = 2000.0 if drone_type == DroneType.LEADER else 1000.0
        self.max_speed = 50.0
        
        self.neighbor_drones = {}
        self.routing_table = {}
        self.route_cache = {}
        
        self.data_queue = deque()
        
        self.packets_forwarded = 0
        self.energy_used = 0.0
        
        self.lock = threading.RLock()
        
    def update_position(self, new_position: Position, time_delta: float = 1.0):
        """Update drone position and handle movement energy consumption"""
        distance_moved = self.position.distance_to(new_position)
        movement_energy = distance_moved * 0.01
        
        with self.lock:
            self.position = new_position
            self.battery_level -= movement_energy
            self.energy_used += movement_energy
            
            if distance_moved > 100.0:
                self.route_cache.clear()
            
    def measure_link_quality(self, neighbor_drone: 'GADrone') -> float:
        """Measure link quality to neighbor drone (0.0 to 1.0)"""
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
    
    def _is_route_valid(self, route: List[str], drone_network: Dict[str, 'GADrone'], 
                       destination: str) -> bool:
        """Validate if a cached route is still valid"""
        if not route or len(route) < 2:
            return False
        
        if route[0] != self.drone_id or route[-1] != destination:
            return False
        
        for i in range(len(route) - 1):
            current_id = route[i]
            next_id = route[i + 1]
            
            if current_id not in drone_network or next_id not in drone_network:
                return False
            
            current_drone = drone_network[current_id]
            if next_id not in current_drone.neighbor_drones:
                return False
        
        return True
        
    def generate_initial_population(self, destination_id: str, drone_network: Dict[str, 'GADrone'], 
                                  population_size: int = 50) -> List[GARouteChromosome]:
        """Generate initial population of routes using random walk"""
        population = []
        max_attempts = population_size * 3
        attempts = 0
        
        while len(population) < population_size and attempts < max_attempts:
            attempts += 1
            route = self._generate_random_route(destination_id, drone_network)
            if route and route not in [chrom.route for chrom in population]:
                chromosome = GARouteChromosome(route, self.drone_id, destination_id)
                population.append(chromosome)
        
        if len(population) < 2:
            if destination_id in self.neighbor_drones:
                direct_route = [self.drone_id, destination_id]
                chromosome = GARouteChromosome(direct_route, self.drone_id, destination_id)
                population.append(chromosome)
            
            for neighbor_id in list(self.neighbor_drones.keys())[:population_size]:
                if neighbor_id != destination_id and neighbor_id in drone_network:
                    neighbor_drone = drone_network[neighbor_id]
                    if destination_id in neighbor_drone.neighbor_drones:
                        two_hop_route = [self.drone_id, neighbor_id, destination_id]
                        chromosome = GARouteChromosome(two_hop_route, self.drone_id, destination_id)
                        population.append(chromosome)
                        if len(population) >= population_size:
                            break
        
        return population
    
    def _generate_random_route(self, destination_id: str, drone_network: Dict[str, 'GADrone'], 
                              max_hops: int = 10) -> Optional[List[str]]:
        """Generate a random route from current drone to destination"""
        route = [self.drone_id]
        current_drone_id = self.drone_id
        visited = {self.drone_id}
        
        for hop in range(max_hops):
            if current_drone_id == destination_id:
                return route
                
            current_drone = drone_network.get(current_drone_id)
            if not current_drone:
                return None
            
            available_neighbors = [
                neighbor_id for neighbor_id in current_drone.neighbor_drones.keys()
                if neighbor_id not in visited and neighbor_id in drone_network
            ]
            
            if destination_id in available_neighbors:
                route.append(destination_id)
                return route
            
            if not available_neighbors:
                return None
            
            next_hop = random.choice(available_neighbors)
            route.append(next_hop)
            visited.add(next_hop)
            current_drone_id = next_hop
            
        return route if route[-1] == destination_id else None
    
    def selection(self, population: List[GARouteChromosome], tournament_size: int = 3) -> List[GARouteChromosome]:
        """Tournament selection with safety checks"""
        if not population:
            return []
        
        valid_population = [chrom for chrom in population if chrom.fitness > 0]
        
        if not valid_population:
            return population
            
        actual_tournament_size = min(tournament_size, len(valid_population))
        selected = []
        
        for _ in range(len(population)):
            if len(valid_population) >= actual_tournament_size and actual_tournament_size > 0:
                tournament = random.sample(valid_population, actual_tournament_size)
                winner = max(tournament, key=lambda chrom: chrom.fitness)
            else:
                winner = max(valid_population, key=lambda chrom: chrom.fitness)
            selected.append(winner)
            
        return selected
    
    def crossover(self, parent1: GARouteChromosome, parent2: GARouteChromosome, 
                 drone_network: Dict[str, 'GADrone']) -> GARouteChromosome:
        """Ordered crossover for routes with fallback"""
        route1, route2 = parent1.route, parent2.route
        
        if len(route1) < 2 or len(route2) < 2:
            return GARouteChromosome(parent1.route.copy(), parent1.source, parent1.destination)
            
        common_nodes = set(route1) & set(route2)
        common_nodes.discard(self.drone_id)
        common_nodes.discard(parent1.destination)
        
        if not common_nodes:
            min_len = min(len(route1), len(route2))
            if min_len > 1:
                crossover_point = random.randint(1, min_len - 1)
                child_route = route1[:crossover_point] + route2[crossover_point:]
            else:
                return GARouteChromosome(parent1.route.copy(), parent1.source, parent1.destination)
        else:
            crossover_point = random.choice(list(common_nodes))
            
            if crossover_point in route1 and crossover_point in route2:
                idx1 = route1.index(crossover_point)
                idx2 = route2.index(crossover_point)
                child_route = route1[:idx1] + route2[idx2:]
            else:
                return GARouteChromosome(parent1.route.copy(), parent1.source, parent1.destination)
        
        seen = set()
        clean_route = []
        for node in child_route:
            if node not in seen:
                clean_route.append(node)
                seen.add(node)
                
        if not clean_route or clean_route[0] != self.drone_id:
            clean_route.insert(0, self.drone_id)
        
        if clean_route[-1] != parent1.destination:
            success = self._complete_route(clean_route, parent1.destination, drone_network)
            if not success:
                return GARouteChromosome(parent1.route.copy(), parent1.source, parent1.destination)
            
        return GARouteChromosome(clean_route, self.drone_id, parent1.destination)
    
    def _complete_route(self, route: List[str], destination: str, 
                       drone_network: Dict[str, 'GADrone']) -> bool:
        """Try to complete an incomplete route"""
        current = route[-1]
        max_attempts = 5
        visited = set(route)
        
        for attempt in range(max_attempts):
            if current == destination:
                return True
                
            current_drone = drone_network.get(current)
            if not current_drone:
                return False
            
            if destination in current_drone.neighbor_drones and destination in drone_network:
                route.append(destination)
                return True
            
            available = [
                neighbor_id for neighbor_id in current_drone.neighbor_drones.keys()
                if neighbor_id not in visited and neighbor_id in drone_network
            ]
            
            if not available:
                return False
            
            next_hop = random.choice(available)
            route.append(next_hop)
            visited.add(next_hop)
            current = next_hop
            
        return current == destination
    
    def mutation(self, chromosome: GARouteChromosome, drone_network: Dict[str, 'GADrone'], 
                 mutation_rate: float = 0.1) -> GARouteChromosome:
        """Swap mutation for routes"""
        if random.random() > mutation_rate or len(chromosome.route) < 3:
            return GARouteChromosome(chromosome.route.copy(), chromosome.source, chromosome.destination)
            
        mutated_route = chromosome.route.copy()
        
        if len(mutated_route) >= 4:
            idx1, idx2 = random.sample(range(1, len(mutated_route) - 1), 2)
            mutated_route[idx1], mutated_route[idx2] = mutated_route[idx2], mutated_route[idx1]
        
        if self._validate_route(mutated_route, drone_network, chromosome.destination):
            return GARouteChromosome(mutated_route, chromosome.source, chromosome.destination)
        else:
            return GARouteChromosome(chromosome.route.copy(), chromosome.source, chromosome.destination)
    
    def _validate_route(self, route: List[str], drone_network: Dict[str, 'GADrone'], 
                       destination: str = None) -> bool:
        """Validate if all consecutive nodes in route are neighbors and route ends at destination"""
        if not route or len(route) < 2:
            return False
        
        if destination and route[-1] != destination:
            return False
        
        for i in range(len(route) - 1):
            current_drone = drone_network.get(route[i])
            if not current_drone or route[i + 1] not in current_drone.neighbor_drones:
                return False
        return True
    
    def find_route_ga(self, destination_id: str, drone_network: Dict[str, 'GADrone'],
                     population_size: int = 30, generations: int = 15, 
                     convergence_threshold: int = 5) -> Optional[List[str]]:
        """Find optimal route using Genetic Algorithm with convergence detection"""
        with self.lock:
            cache_key = f"{self.drone_id}_{destination_id}"
            if cache_key in self.route_cache:
                cached_route, timestamp = self.route_cache[cache_key]
                if (time.time() - timestamp < 30.0 and 
                    self._is_route_valid(cached_route, drone_network, destination_id)):
                    return cached_route
                else:
                    del self.route_cache[cache_key]
        
        population = self.generate_initial_population(destination_id, drone_network, population_size)
        
        if not population:
            return None
            
        for chrom in population:
            chrom.calculate_fitness(drone_network)
        
        valid_population = [chrom for chrom in population if chrom.fitness > 0]
        
        if not valid_population:
            return None
        
        best_chromosome = max(valid_population, key=lambda chrom: chrom.fitness)
        
        generations_without_improvement = 0
        
        if len(valid_population) >= 2:
            for generation in range(generations):
                selected = self.selection(valid_population)
                
                if not selected or len(selected) < 2:
                    break
                    
                new_population = []
                new_population.append(GARouteChromosome(best_chromosome.route.copy(), 
                                                       best_chromosome.source, 
                                                       best_chromosome.destination))
                new_population[0].fitness = best_chromosome.fitness
                
                while len(new_population) < population_size:
                    if len(selected) < 2:
                        new_chrom = GARouteChromosome(best_chromosome.route.copy(),
                                                     best_chromosome.source,
                                                     best_chromosome.destination)
                        new_chrom.fitness = best_chromosome.fitness
                        new_population.append(new_chrom)
                    else:
                        parent1, parent2 = random.sample(selected, 2)
                        child = self.crossover(parent1, parent2, drone_network)
                        child = self.mutation(child, drone_network, 0.1)
                        child.calculate_fitness(drone_network)
                        new_population.append(child)
                
                valid_population = [chrom for chrom in new_population if chrom.fitness > 0]
                
                if not valid_population:
                    valid_population = [best_chromosome]
                    break
                
                current_best = max(valid_population, key=lambda chrom: chrom.fitness)
                if current_best.fitness > best_chromosome.fitness:
                    best_chromosome = current_best
                    generations_without_improvement = 0
                else:
                    generations_without_improvement += 1
                
                if generations_without_improvement >= convergence_threshold:
                    break
        
        if best_chromosome.fitness > 0.1:
            with self.lock:
                self.route_cache[cache_key] = (best_chromosome.route, time.time())
            return best_chromosome.route
        
        return None
    
    def route_data_packet(self, packet: Dict, drone_network: Dict[str, 'GADrone']) -> Optional[str]:
        """Route data packet using GA-based routing"""
        with self.lock:
            destination = packet.get('destination')
            if destination == self.drone_id:
                return None
            
            if destination in self.neighbor_drones:
                self.packets_forwarded += 1
                transmission_energy = 0.05
                self.battery_level -= transmission_energy
                self.energy_used += transmission_energy
                return destination
                
            route = self.find_route_ga(destination, drone_network)
            if route and len(route) > 1 and route[0] == self.drone_id:
                self.packets_forwarded += 1
                transmission_energy = 0.05
                self.battery_level -= transmission_energy
                self.energy_used += transmission_energy
                return route[1]
            
            return None

    def add_neighbor(self, neighbor_id: str, link_metrics: LinkMetrics):
        """Add or update neighbor drone information"""
        with self.lock:
            self.neighbor_drones[neighbor_id] = link_metrics

    def remove_neighbor(self, neighbor_id: str):
        """Remove neighbor drone"""
        with self.lock:
            if neighbor_id in self.neighbor_drones:
                del self.neighbor_drones[neighbor_id]
            self.route_cache.clear()

    def get_status(self) -> Dict:
        """Get current drone status"""
        return {
            'drone_id': self.drone_id,
            'position': (self.position.x, self.position.y, self.position.z),
            'battery_level': self.battery_level,
            'neighbors_count': len(self.neighbor_drones),
            'packets_forwarded': self.packets_forwarded,
            'energy_used': self.energy_used
        }

class DroneSwarmGAController:
    def __init__(self):
        self.drones: Dict[str, GADrone] = {}
        self.mission_type = MissionType.SURVEILLANCE
        self.performance_metrics = {
            'packet_delivery_rate': [],
            'average_latency': [],
            'energy_efficiency': [],
            'route_discovery_time': []
        }
        
        self.ga_params = {
            'population_size': 30,
            'generations': 15,
            'mutation_rate': 0.1,
            'tournament_size': 3,
            'convergence_threshold': 5
        }
        
        self.running = False
        self.simulation_time = 0

    def initialize_swarm(self, num_drones: int, area_size: Tuple[float, float, float] = (5000, 5000, 500)):
        """Initialize drone swarm in the specified area"""
        print(f"Initializing GA-based swarm with {num_drones} drones...")
        
        num_leaders = max(1, num_drones // 10)
        for i in range(num_leaders):
            position = Position(
                x=random.uniform(0, area_size[0]),
                y=random.uniform(0, area_size[1]),
                z=random.uniform(100, area_size[2])
            )
            drone_id = f"leader_{i}"
            self.drones[drone_id] = GADrone(drone_id, position, DroneType.LEADER, 150.0)
        
        for i in range(num_drones - num_leaders):
            position = Position(
                x=random.uniform(0, area_size[0]),
                y=random.uniform(0, area_size[1]),
                z=random.uniform(50, area_size[2])
            )
            drone_id = f"worker_{i}"
            self.drones[drone_id] = GADrone(drone_id, position, DroneType.WORKER, 100.0)
        
        self.update_neighbor_relationships()
        print("GA-based swarm initialization completed!")

    def update_neighbor_relationships(self):
        """Update neighbor relationships based on current positions"""
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
        """Update drone positions based on mission type"""
        for drone in self.drones.values():
            if self.mission_type == MissionType.SURVEILLANCE:
                new_position = Position(
                    x=drone.position.x + random.uniform(-50, 50),
                    y=drone.position.y + random.uniform(-50, 50),
                    z=drone.position.z + random.uniform(-10, 10)
                )
            elif self.mission_type == MissionType.DELIVERY:
                new_position = Position(
                    x=drone.position.x + random.uniform(-20, 100),
                    y=drone.position.y + random.uniform(-20, 100),
                    z=drone.position.z
                )
            else:
                new_position = Position(
                    x=drone.position.x + random.uniform(-30, 30),
                    y=drone.position.y + random.uniform(-30, 30),
                    z=drone.position.z + random.uniform(-5, 5)
                )
            
            new_position.x = max(0, min(5000, new_position.x))
            new_position.y = max(0, min(5000, new_position.y))
            new_position.z = max(10, min(500, new_position.z))
            
            drone.update_position(new_position)

    def simulate_data_traffic(self):
        """Simulate data traffic between drones"""
        if len(self.drones) < 2:
            return
            
        drone_ids = list(self.drones.keys())
        num_packets = min(3, len(drone_ids) // 4)
        
        for _ in range(num_packets):
            source_id = random.choice(drone_ids)
            destination_id = random.choice([did for did in drone_ids if did != source_id])
            
            packet = {
                'source': source_id,
                'destination': destination_id,
                'size': random.randint(100, 5000),
                'timestamp': time.time(),
                'priority': random.choice(['low', 'medium', 'high'])
            }
            
            self.route_data_packet(packet)

    def route_data_packet(self, packet: Dict):
        """Route data packet through the network"""
        source_id = packet['source']
        destination_id = packet['destination']
        
        if source_id not in self.drones:
            return
            
        current_drone = self.drones[source_id]
        path = [source_id]
        max_hops = 10
        hop_count = 0
        visited = {source_id}
        
        while hop_count < max_hops:
            next_hop_id = current_drone.route_data_packet(packet, self.drones)
            
            if not next_hop_id or next_hop_id not in self.drones:
                break
            
            if next_hop_id in visited:
                break
            
            current_drone = self.drones[next_hop_id]
            path.append(next_hop_id)
            visited.add(next_hop_id)
            hop_count += 1
            
            if next_hop_id == destination_id:
                latency = time.time() - packet['timestamp']
                self.performance_metrics['average_latency'].append(latency)
                break

    def run_simulation(self, duration: int = 60):
        """Main simulation loop"""
        print("Starting Drone Swarm GA Simulation...")
        self.running = True
        start_time = time.time()
        
        try:
            while self.running and (time.time() - start_time) < duration:
                loop_start = time.time()
                self.simulation_time += 1
                
                self.update_drone_positions()
                self.update_neighbor_relationships()
                self.simulate_data_traffic()
                
                if self.simulation_time % 10 == 0:
                    self._print_status()
                
                loop_time = time.time() - loop_start
                time.sleep(max(0, 1.0 - loop_time))
                
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        except Exception as e:
            print(f"\nSimulation error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            self._print_final_stats()
            # Visualize the network after simulation ends
            self.visualize_network()

    def _print_status(self):
        """Print current network status"""
        total_drones = len(self.drones)
        if total_drones == 0:
            return
            
        avg_battery = sum(drone.battery_level for drone in self.drones.values()) / total_drones
        total_packets = sum(drone.packets_forwarded for drone in self.drones.values())
        
        print(f"\n--- GA Network Status at Time {self.simulation_time} ---")
        print(f"Active Drones: {total_drones}")
        print(f"Average Battery: {avg_battery:.1f}%")
        print(f"Total Packets Forwarded: {total_packets}")

    def _print_final_stats(self):
        """Print final simulation statistics"""
        print("\n" + "="*50)
        print("GA SIMULATION FINAL STATISTICS")
        print("="*50)
        
        total_drones = len(self.drones)
        if total_drones == 0:
            return
            
        total_packets = sum(drone.packets_forwarded for drone in self.drones.values())
        total_energy = sum(drone.energy_used for drone in self.drones.values())
        avg_battery = sum(drone.battery_level for drone in self.drones.values()) / total_drones
        
        print(f"Total Drones: {total_drones}")
        print(f"Total Packets Forwarded: {total_packets}")
        print(f"Total Energy Used: {total_energy:.2f}")
        print(f"Average Battery Remaining: {avg_battery:.1f}%")
        
        if self.performance_metrics['average_latency']:
            avg_latency = np.mean(self.performance_metrics['average_latency'])
            print(f"Average Packet Latency: {avg_latency:.2f}s")
    
    def visualize_network(self, save_path: str = "drone_network.png"):
        """Visualize the drone network in 3D"""
        if not self.drones:
            print("No drones to visualize")
            return
        
        fig = plt.figure(figsize=(15, 10))
        
        # 3D Network visualization
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Separate leaders and workers
        leader_positions = []
        worker_positions = []
        leader_battery = []
        worker_battery = []
        
        for drone in self.drones.values():
            pos = (drone.position.x, drone.position.y, drone.position.z)
            if drone.drone_type == DroneType.LEADER:
                leader_positions.append(pos)
                leader_battery.append(drone.battery_level)
            else:
                worker_positions.append(pos)
                worker_battery.append(drone.battery_level)
        
        # Plot drones
        if leader_positions:
            leader_positions = np.array(leader_positions)
            ax1.scatter(leader_positions[:, 0], leader_positions[:, 1], leader_positions[:, 2],
                       c=leader_battery, cmap='Reds', s=200, marker='^', 
                       label='Leaders', edgecolors='black', linewidth=2, alpha=0.8)
        
        if worker_positions:
            worker_positions = np.array(worker_positions)
            ax1.scatter(worker_positions[:, 0], worker_positions[:, 1], worker_positions[:, 2],
                       c=worker_battery, cmap='Blues', s=100, marker='o', 
                       label='Workers', edgecolors='black', linewidth=1, alpha=0.8)
        
        # Draw connections between neighbors
        for drone_id, drone in self.drones.items():
            for neighbor_id in drone.neighbor_drones.keys():
                if neighbor_id in self.drones:
                    neighbor = self.drones[neighbor_id]
                    ax1.plot([drone.position.x, neighbor.position.x],
                            [drone.position.y, neighbor.position.y],
                            [drone.position.z, neighbor.position.z],
                            'gray', alpha=0.2, linewidth=0.5)
        
        ax1.set_xlabel('X Position (m)', fontsize=10)
        ax1.set_ylabel('Y Position (m)', fontsize=10)
        ax1.set_zlabel('Z Position (m)', fontsize=10)
        ax1.set_title('3D Drone Network Topology', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2D Top-down view with statistics
        ax2 = fig.add_subplot(122)
        
        # Plot 2D projection
        if leader_positions.size > 0:
            ax2.scatter(leader_positions[:, 0], leader_positions[:, 1],
                       c=leader_battery, cmap='Reds', s=200, marker='^', 
                       label='Leaders', edgecolors='black', linewidth=2, alpha=0.8)
        
        if worker_positions.size > 0:
            ax2.scatter(worker_positions[:, 0], worker_positions[:, 1],
                       c=worker_battery, cmap='Blues', s=100, marker='o', 
                       label='Workers', edgecolors='black', linewidth=1, alpha=0.8)
        
        # Draw connections in 2D
        for drone_id, drone in self.drones.items():
            for neighbor_id in drone.neighbor_drones.keys():
                if neighbor_id in self.drones:
                    neighbor = self.drones[neighbor_id]
                    ax2.plot([drone.position.x, neighbor.position.x],
                            [drone.position.y, neighbor.position.y],
                            'gray', alpha=0.2, linewidth=0.5)
        
        ax2.set_xlabel('X Position (m)', fontsize=10)
        ax2.set_ylabel('Y Position (m)', fontsize=10)
        ax2.set_title('Top-Down Network View', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal')
        
        # Add statistics text box
        total_drones = len(self.drones)
        total_packets = sum(drone.packets_forwarded for drone in self.drones.values())
        total_energy = sum(drone.energy_used for drone in self.drones.values())
        avg_battery = sum(drone.battery_level for drone in self.drones.values()) / total_drones
        avg_neighbors = sum(len(drone.neighbor_drones) for drone in self.drones.values()) / total_drones
        
        stats_text = f"""Network Statistics:
Total Drones: {total_drones}
Leaders: {len(leader_positions)}
Workers: {len(worker_positions)}
Avg Battery: {avg_battery:.1f}%
Total Packets: {total_packets}
Total Energy: {total_energy:.2f}
Avg Neighbors: {avg_neighbors:.1f}
Simulation Time: {self.simulation_time}s"""
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.colorbar(ax2.collections[0] if worker_positions.size > 0 else ax2.collections[-1], 
                     ax=ax2, label='Battery Level (%)')
        
        plt.suptitle('GA-based Drone Swarm Network Visualization', 
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nNetwork visualization saved to: {save_path}")
        
        # Show the plot
        plt.show()

if __name__ == "__main__":
    controller = DroneSwarmGAController()
    controller.initialize_swarm(25, area_size=(2000, 2000, 300))
    print("Starting 1-minute GA simulation...")
    controller.run_simulation(duration=60)
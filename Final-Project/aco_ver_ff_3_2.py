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
import networkx as nx

RNG = 4136314
random.seed(RNG)
np.random.seed(RNG)

class DroneType(Enum):
    LEADER = "leader"
    WORKER = "worker"
    RELAY = "relay"

class AntType(Enum):
    EXPLORATORY = "exploratory"
    DATA_COLLECTION = "data_collection"
    EMERGENCY = "emergency"

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

class ForwardAnt:
    def __init__(self, ant_id: str, source_drone: 'IntelligentDrone', destination_drone: 'IntelligentDrone', ant_type: AntType = AntType.EXPLORATORY):
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
        self.emergency_flag = False
        self.timestamp = time.time()
        
    def calculate_path_metrics(self) -> Dict[str, float]:
        if not self.forward_ant.path_taken:
            return {}
            
        total_quality = 0.0
        for hop in self.forward_ant.path_taken:
            total_quality += hop['link_quality']
            
        avg_quality = total_quality / len(self.forward_ant.path_taken)
        hop_penalty = len(self.forward_ant.path_taken) * 0.1
        
        self.path_quality_score = avg_quality - hop_penalty
        return {
            'quality_score': self.path_quality_score,
            'hop_count': len(self.forward_ant.path_taken),
            'avg_quality': avg_quality
        }
        
    def update_pheromones(self, drone_network: Dict[str, 'IntelligentDrone']):
        metrics = self.calculate_path_metrics()
        path_quality = max(0.1, metrics['quality_score'])
        
        for i in range(len(self.forward_ant.path_taken) - 1):
            current_drone_id = self.forward_ant.path_taken[i]['drone_id']
            next_drone_id = self.forward_ant.path_taken[i + 1]['drone_id']
            
            if current_drone_id in drone_network and next_drone_id in drone_network:
                current_drone = drone_network[current_drone_id]
                current_drone.update_pheromone(next_drone_id, path_quality)

class IntelligentDrone:
    def __init__(self, drone_id: str, position: Position, drone_type: DroneType, 
                 initial_energy: float = 100.0):
        self.drone_id = drone_id
        self.position = position
        self.drone_type = drone_type
        self.battery_level = initial_energy
        self.initial_energy = initial_energy
        self.communication_range = 2000.0 if drone_type == DroneType.LEADER else 1000.0
        self.max_speed = 50.0
        self.current_mission = None
        
        self.pheromone_table = {}
        self.neighbor_drones = {}
        self.routing_table = {}
        
        self.ant_queue = deque()
        self.data_queue = deque()
        
        self.packets_forwarded = 0
        self.ants_processed = 0
        self.energy_used = 0.0
        
        self.lock = threading.RLock()
        
    def update_position(self, new_position: Position, time_delta: float = 1.0):
        distance_moved = self.position.distance_to(new_position)
        movement_energy = distance_moved * 0.01
        
        with self.lock:
            self.position = new_position
            self.battery_level -= movement_energy
            self.energy_used += movement_energy
            
    def measure_link_quality(self, neighbor_drone: 'IntelligentDrone') -> float:
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
        
    def calculate_heuristic(self, neighbor_drone: 'IntelligentDrone', destination: 'IntelligentDrone') -> float:
        los_quality = self.measure_link_quality(neighbor_drone)
        distance_to_neighbor = self.position.distance_to(neighbor_drone.position)
        signal_strength = 1.0 - (distance_to_neighbor / self.communication_range)
        battery_compatibility = min(self.battery_level, neighbor_drone.battery_level) / 100.0
        mobility_sync = self.predict_link_duration(neighbor_drone)
        
        current_to_dest = self.position.distance_to(destination.position)
        neighbor_to_dest = neighbor_drone.position.distance_to(destination.position)
        progress = 1.0 if neighbor_to_dest < current_to_dest else 0.5
        
        heuristic = (0.30 * los_quality + 
                    0.25 * signal_strength + 
                    0.20 * battery_compatibility + 
                    0.15 * mobility_sync + 
                    0.10 * progress)
        
        return max(0.1, heuristic)
        
    def predict_link_duration(self, neighbor_drone: 'IntelligentDrone') -> float:
        return 0.8
        
    def process_forward_ant(self, ant: ForwardAnt) -> Optional[str]:
        with self.lock:
            self.ants_processed += 1
            link_quality = 1.0
            ant.record_hop(self.drone_id, link_quality, self.position)
            
            if self.drone_id == ant.destination_drone.drone_id:
                return None
                
            if not ant.should_continue():
                return None
                
            available_neighbors = self.get_available_neighbors(ant)
            if not available_neighbors:
                return None
                
            next_hop_id = self.probabilistic_routing_decision(available_neighbors, ant.destination_drone)
            return next_hop_id
            
    def get_available_neighbors(self, ant: ForwardAnt) -> List['IntelligentDrone']:
        available = []
        visited_drones = {hop['drone_id'] for hop in ant.path_taken}
        
        for neighbor_id, metrics in self.neighbor_drones.items():
            if neighbor_id not in visited_drones:
                available.append(neighbor_id)
                
        return available
        
    def probabilistic_routing_decision(self, available_neighbors: List[str], destination: 'IntelligentDrone') -> str:
        if not available_neighbors:
            return random.choice(available_neighbors) if available_neighbors else None
            
        probabilities = []
        total = 0.0
        
        alpha = 0.6
        beta = 0.4
        
        for neighbor_id in available_neighbors:
            if destination.drone_id not in self.pheromone_table:
                self.pheromone_table[destination.drone_id] = {}
            if neighbor_id not in self.pheromone_table[destination.drone_id]:
                self.pheromone_table[destination.drone_id][neighbor_id] = 0.1
                
            pheromone = self.pheromone_table[destination.drone_id][neighbor_id]
            heuristic = 0.7
            
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
        
    def process_backward_ant(self, ant: BackwardAnt):
        with self.lock:
            if ant.path_quality_score > 0.5:
                if ant.forward_ant.path_taken:
                    first_hop = ant.forward_ant.path_taken[0]
                    if len(ant.forward_ant.path_taken) > 1:
                        next_hop = ant.forward_ant.path_taken[1]['drone_id']
                        self.routing_table[ant.forward_ant.destination_drone.drone_id] = next_hop

    def update_pheromone(self, neighbor_id: str, path_quality: float):
        evaporation_rate = 0.3
        Q = 2.0
        
        with self.lock:
            if neighbor_id not in self.pheromone_table:
                self.pheromone_table[neighbor_id] = {}
            
            current_pheromone = self.pheromone_table.get(neighbor_id, {}).get('default', 0.1)
            evaporated_pheromone = current_pheromone * (1 - evaporation_rate)
            reinforcement = Q * path_quality
            new_pheromone = evaporated_pheromone + reinforcement
            new_pheromone = max(0.1, min(1.0, new_pheromone))
            
            if 'default' not in self.pheromone_table[neighbor_id]:
                self.pheromone_table[neighbor_id]['default'] = new_pheromone
            else:
                self.pheromone_table[neighbor_id]['default'] = new_pheromone

    def route_data_packet(self, packet: Dict) -> Optional[str]:
        with self.lock:
            destination = packet.get('destination')
            if destination in self.routing_table:
                self.packets_forwarded += 1
                transmission_energy = 0.05
                self.battery_level -= transmission_energy
                self.energy_used += transmission_energy
                return self.routing_table[destination]
            return None

    def add_neighbor(self, neighbor_id: str, link_metrics: LinkMetrics):
        with self.lock:
            self.neighbor_drones[neighbor_id] = link_metrics

    def remove_neighbor(self, neighbor_id: str):
        with self.lock:
            if neighbor_id in self.neighbor_drones:
                del self.neighbor_drones[neighbor_id]

    def get_status(self) -> Dict:
        return {
            'drone_id': self.drone_id,
            'position': (self.position.x, self.position.y, self.position.z),
            'battery_level': self.battery_level,
            'neighbors_count': len(self.neighbor_drones),
            'packets_forwarded': self.packets_forwarded,
            'ants_processed': self.ants_processed,
            'energy_used': self.energy_used
        }

class DroneSwarmACOController:
    def __init__(self):
        self.drones: Dict[str, IntelligentDrone] = {}
        self.mission_type = MissionType.SURVEILLANCE
        self.performance_metrics = {
            'packet_delivery_rate': [],
            'average_latency': [],
            'energy_efficiency': [],
            'route_discovery_time': []
        }
        
        self.aco_params = {
            'alpha': 0.6,
            'beta': 0.4,
            'rho': 0.3,
            'Q': 2.0,
            'ant_generation_interval': 5.0,
            'max_ants_per_second': 10
        }
        
        self.environment = {
            'obstacles': [],
            'weather_conditions': 'clear',
            'interference_level': 0.1
        }
        
        self.running = False
        self.simulation_time = 0
        self.ant_counter = 0
        self.emergency_mode = False
        
        self.ant_thread = None
        self.monitor_thread = None
        self.lock = threading.RLock()

    def initialize_swarm(self, num_drones: int, area_size: Tuple[float, float, float] = (5000, 5000, 500)):
        print(f"Initializing swarm with {num_drones} drones...")
        
        num_leaders = max(1, num_drones // 10)
        for i in range(num_leaders):
            position = Position(
                x=random.uniform(0, area_size[0]),
                y=random.uniform(0, area_size[1]),
                z=random.uniform(100, area_size[2])
            )
            drone_id = f"leader_{i}"
            self.drones[drone_id] = IntelligentDrone(drone_id, position, DroneType.LEADER, 150.0)
        
        for i in range(num_drones - num_leaders):
            position = Position(
                x=random.uniform(0, area_size[0]),
                y=random.uniform(0, area_size[1]),
                z=random.uniform(50, area_size[2])
            )
            drone_id = f"worker_{i}"
            self.drones[drone_id] = IntelligentDrone(drone_id, position, DroneType.WORKER, 100.0)
        
        self.update_neighbor_relationships()
        print("Swarm initialization completed!")

    def add_drone_manual(self, x: float, y: float, z: float, drone_type: DroneType):
        """Add a drone manually at specific coordinates - FIXED"""
        if drone_type == DroneType.LEADER:
            existing_count = sum(1 for d in self.drones.values() if d.drone_type == DroneType.LEADER)
            drone_id = f"leader_{existing_count}"
            initial_energy = 150.0
        else:
            existing_count = sum(1 for d in self.drones.values() if d.drone_type == DroneType.WORKER)
            drone_id = f"worker_{existing_count}"
            initial_energy = 100.0
        
        position = Position(x, y, z)
        new_drone = IntelligentDrone(drone_id, position, drone_type, initial_energy)
        self.drones[drone_id] = new_drone
        print(f"Added {drone_type.value.upper()} drone '{drone_id}' at ({x:.0f}, {y:.0f}, {z:.0f})")

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

    def launch_ant_agents(self):
        if not self.drones:
            return
            
        drone_ids = list(self.drones.keys())
        num_ants = min(self.aco_params['max_ants_per_second'], len(drone_ids) // 2)
        
        for _ in range(num_ants):
            source_id = random.choice(drone_ids)
            destination_id = random.choice([did for did in drone_ids if did != source_id])
            
            source_drone = self.drones[source_id]
            destination_drone = self.drones[destination_id]
            
            ant_id = f"ant_{self.ant_counter}"
            self.ant_counter += 1
            
            forward_ant = ForwardAnt(ant_id, source_drone, destination_drone)
            self.process_forward_ant(forward_ant)

    def process_forward_ant(self, ant: ForwardAnt):
        current_drone = ant.source_drone
        start_time = time.time()
        max_processing_time = 10.0
        
        while current_drone and ant.should_continue() and (time.time() - start_time) < max_processing_time:
            next_hop_id = current_drone.process_forward_ant(ant)
            
            if not next_hop_id:
                break
                
            if next_hop_id in self.drones:
                current_drone = self.drones[next_hop_id]
            else:
                break
        
        if ant.path_taken and ant.path_taken[-1]['drone_id'] == ant.destination_drone.drone_id:
            backward_ant = BackwardAnt(ant)
            self.process_backward_ant(backward_ant)

    def process_backward_ant(self, ant: BackwardAnt):
        ant.calculate_path_metrics()
        ant.update_pheromones(self.drones)
        
        route_discovery_time = time.time() - ant.forward_ant.timestamp
        self.performance_metrics['route_discovery_time'].append(route_discovery_time)

    def update_drone_positions(self):
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
        if len(self.drones) < 2:
            return
            
        drone_ids = list(self.drones.keys())
        num_packets = random.randint(1, len(drone_ids) // 4)
        
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
        source_id = packet['source']
        destination_id = packet['destination']
        
        if source_id not in self.drones:
            return
            
        current_drone = self.drones[source_id]
        path = [source_id]
        max_hops = 15
        hop_count = 0
        
        while hop_count < max_hops:
            next_hop_id = current_drone.route_data_packet(packet)
            
            if not next_hop_id or next_hop_id not in self.drones:
                break
                
            current_drone = self.drones[next_hop_id]
            path.append(next_hop_id)
            hop_count += 1
            
            if next_hop_id == destination_id:
                latency = time.time() - packet['timestamp']
                self.performance_metrics['average_latency'].append(latency)
                break
        
        return path

    def adaptive_parameter_tuning(self):
        total_drones = len(self.drones)
        if total_drones == 0:
            return
            
        avg_neighbors = sum(len(drone.neighbor_drones) for drone in self.drones.values()) / total_drones
        network_density = avg_neighbors / total_drones
        
        if network_density > 0.3:
            self.aco_params['alpha'] = 0.7
            self.aco_params['beta'] = 0.3
            self.aco_params['rho'] = 0.4
        elif network_density < 0.1:
            self.aco_params['alpha'] = 0.4
            self.aco_params['beta'] = 0.6
            self.aco_params['rho'] = 0.2
        else:
            self.aco_params['alpha'] = 0.6
            self.aco_params['beta'] = 0.4
            self.aco_params['rho'] = 0.3

    def monitor_network_health(self):
        total_drones = len(self.drones)
        if total_drones == 0:
            return
            
        avg_battery = sum(drone.battery_level for drone in self.drones.values()) / total_drones
        total_packets = sum(drone.packets_forwarded for drone in self.drones.values())
        total_ants = sum(drone.ants_processed for drone in self.drones.values())
        
        if self.simulation_time % 10 == 0 and self.running:
            print(f"\n--- Network Status at Time {self.simulation_time} ---")
            print(f"Active Drones: {total_drones}")
            print(f"Average Battery: {avg_battery:.1f}%")
            print(f"Total Packets Forwarded: {total_packets}")
            print(f"Total Ants Processed: {total_ants}")

    def visualize_network(self):
        """Visualize network with drone names - FIXED"""
        try:
            plt.figure(figsize=(14, 10))
            
            for drone_id, drone in self.drones.items():
                color = 'red' if drone.drone_type == DroneType.LEADER else 'blue'
                marker = 's' if drone.drone_type == DroneType.LEADER else 'o'
                size = 150 if drone.drone_type == DroneType.LEADER else 100
                
                plt.scatter(drone.position.x, drone.position.y, c=color, 
                          marker=marker, s=size, alpha=0.7, edgecolors='black', linewidth=1.5)
                
                # Drone name label
                plt.text(drone.position.x, drone.position.y + 80, 
                        drone_id, fontsize=9, ha='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black'))
                
                # Battery level
                plt.text(drone.position.x, drone.position.y - 80, 
                        f"{drone.battery_level:.0f}%", fontsize=8, ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.7))
                
                for neighbor_id in drone.neighbor_drones.keys():
                    if neighbor_id in self.drones:
                        neighbor = self.drones[neighbor_id]
                        plt.plot([drone.position.x, neighbor.position.x],
                                [drone.position.y, neighbor.position.y], 'gray', alpha=0.3, linewidth=1)
            
            leader_count = sum(1 for d in self.drones.values() if d.drone_type == DroneType.LEADER)
            worker_count = sum(1 for d in self.drones.values() if d.drone_type == DroneType.WORKER)
            
            title = f"Drone Swarm Network - Time: {self.simulation_time}s\n"
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
            
        except Exception as e:
            print(f"Visualization error: {e}")

    def test_routing_between_drones(self, source_id: Optional[str] = None, 
                                    destination_id: Optional[str] = None):
        """Test and visualize routing between two specific drones"""
        if len(self.drones) < 2:
            print("ERROR: Need at least 2 drones for routing test!")
            return
        
        drone_ids = list(self.drones.keys())
        
        if source_id is None:
            source_id = random.choice(drone_ids)
        if destination_id is None:
            destination_id = random.choice([did for did in drone_ids if did != source_id])
        
        if source_id not in self.drones or destination_id not in self.drones:
            print(f"ERROR: Invalid drone IDs")
            return
        
        print(f"\nTesting route from {source_id} to {destination_id}...")
        
        self.visualize_routing_path(source_id, destination_id)

    def visualize_routing_path(self, source_id: str, destination_id: str, show_all_connections: bool = True):
        """Visualize the routing path from source to destination drone"""
        if source_id not in self.drones or destination_id not in self.drones:
            print(f"ERROR: Source or destination drone not found!")
            return
        
        packet = {
            'source': source_id,
            'destination': destination_id,
            'size': 1000,
            'timestamp': time.time(),
            'priority': 'high'
        }
        
        path = self.route_data_packet(packet)
        
        if not path:
            print(f"No route found from {source_id} to {destination_id}")
            return
        
        plt.figure(figsize=(14, 10))
        ax = plt.gca()
        
        for drone_id, drone in self.drones.items():
            if drone.drone_type == DroneType.LEADER:
                color = 'red'
                marker = 's'
                size = 150
                alpha = 0.3
            else:
                color = 'blue'
                marker = 'o'
                size = 100
                alpha = 0.3
            
            if drone_id == source_id:
                color = 'green'
                alpha = 1.0
                size = 200
            elif drone_id == destination_id:
                color = 'orange'
                alpha = 1.0
                size = 200
            elif drone_id in path:
                alpha = 0.8
            
            plt.scatter(drone.position.x, drone.position.y, 
                       c=color, marker=marker, s=size, alpha=alpha,
                       edgecolors='black', linewidth=1.5, zorder=3)
            
            label_color = 'black' if drone_id in path or drone_id in [source_id, destination_id] else 'gray'
            fontsize = 10 if drone_id in path or drone_id in [source_id, destination_id] else 8
            fontweight = 'bold' if drone_id in path or drone_id in [source_id, destination_id] else 'normal'
            
            plt.text(drone.position.x, drone.position.y + 80, 
                    drone_id, fontsize=fontsize, ha='center',
                    color=label_color, fontweight=fontweight,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        if show_all_connections:
            for drone_id, drone in self.drones.items():
                for neighbor_id in drone.neighbor_drones.keys():
                    if neighbor_id in self.drones:
                        neighbor = self.drones[neighbor_id]
                        plt.plot([drone.position.x, neighbor.position.x],
                               [drone.position.y, neighbor.position.y], 
                               'gray', alpha=0.15, linewidth=0.5, zorder=1)
        
        for i in range(len(path) - 1):
            current_drone = self.drones[path[i]]
            next_drone = self.drones[path[i + 1]]
            
            plt.plot([current_drone.position.x, next_drone.position.x],
                    [current_drone.position.y, next_drone.position.y],
                    'purple', linewidth=4, alpha=0.7, zorder=2)
            
            dx = next_drone.position.x - current_drone.position.x
            dy = next_drone.position.y - current_drone.position.y
            plt.arrow(current_drone.position.x, current_drone.position.y,
                     dx * 0.8, dy * 0.8,
                     head_width=50, head_length=50, fc='purple', 
                     ec='purple', alpha=0.7, zorder=2, linewidth=2)
            
            mid_x = (current_drone.position.x + next_drone.position.x) / 2
            mid_y = (current_drone.position.y + next_drone.position.y) / 2
            plt.text(mid_x, mid_y, f"Hop {i+1}", fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8),
                    ha='center', fontweight='bold', zorder=4)
        
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=f'Source: {source_id}',
                   markerfacecolor='green', markersize=12),
            Line2D([0], [0], marker='o', color='w', label=f'Destination: {destination_id}',
                   markerfacecolor='orange', markersize=12),
            Line2D([0], [0], color='purple', linewidth=4, label='Routing Path'),
            Line2D([0], [0], marker='s', color='w', label='Leader Drone',
                   markerfacecolor='red', markersize=10, alpha=0.7),
            Line2D([0], [0], marker='o', color='w', label='Worker Drone',
                   markerfacecolor='blue', markersize=8, alpha=0.7)
        ]
        
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        path_success = "SUCCESS" if path[-1] == destination_id else "FAILED"
        title = f"Routing Path Visualization - {path_success}\n"
        title += f"Source: {source_id} → Destination: {destination_id}\n"
        title += f"Hops: {len(path)-1} | Path: {' → '.join(path)}"
        plt.title(title, fontsize=12, fontweight='bold')
        
        plt.xlabel("X Position (m)", fontsize=11)
        plt.ylabel("Y Position (m)", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*60)
        print("ROUTING PATH DETAILS")
        print("="*60)
        print(f"Source: {source_id}")
        print(f"Destination: {destination_id}")
        print(f"Total Hops: {len(path)-1}")
        print(f"Path Taken: {' → '.join(path)}")
        
        if path[-1] == destination_id:
            print(f"Status: ✓ Packet delivered successfully!")
        else:
            print(f"Status: ✗ Packet delivery failed (no route)")
        
        print("="*60 + "\n")

    def run_simulation(self, duration: int = 300):
        print("Starting Drone Swarm ACO Simulation...")
        self.running = True
        start_time = time.time()
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.ant_thread = threading.Thread(target=self._ant_generation_loop, daemon=True)
        self.ant_thread.start()
        
        try:
            while self.running and (time.time() - start_time) < duration:
                loop_start = time.time()
                
                self.simulation_time += 1
                self.update_drone_positions()
                self.update_neighbor_relationships()
                self.simulate_data_traffic()
                self.adaptive_parameter_tuning()
                self.monitor_network_health()
                
                loop_time = time.time() - loop_start
                sleep_time = max(0, 1.0 - loop_time)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        finally:
            self.running = False
            self._print_final_stats()

    def _monitoring_loop(self):
        while self.running:
            time.sleep(5)

    def _ant_generation_loop(self):
        while self.running:
            self.launch_ant_agents()
            time.sleep(self.aco_params['ant_generation_interval'])

    def _print_final_stats(self):
        print("\n" + "="*50)
        print("FINAL SIMULATION STATISTICS")
        print("="*50)
        
        total_drones = len(self.drones)
        if total_drones == 0:
            return
            
        total_packets = sum(drone.packets_forwarded for drone in self.drones.values())
        total_ants = sum(drone.ants_processed for drone in self.drones.values())
        total_energy = sum(drone.energy_used for drone in self.drones.values())
        avg_battery = sum(drone.battery_level for drone in self.drones.values()) / total_drones
        
        leaders = [d for d in self.drones.values() if d.drone_type == DroneType.LEADER]
        workers = [d for d in self.drones.values() if d.drone_type == DroneType.WORKER]
        
        print(f"Total Drones: {total_drones}")
        print(f"  - Leaders: {len(leaders)}")
        print(f"  - Workers: {len(workers)}")
        print(f"Total Packets Forwarded: {total_packets}")
        print(f"Total Ants Processed: {total_ants}")
        print(f"Total Energy Used: {total_energy:.2f}")
        print(f"Average Battery Remaining: {avg_battery:.1f}%")
        
        if self.performance_metrics['route_discovery_time']:
            avg_discovery_time = np.mean(self.performance_metrics['route_discovery_time'])
            print(f"Average Route Discovery Time: {avg_discovery_time:.2f}s")
        
        if self.performance_metrics['average_latency']:
            avg_latency = np.mean(self.performance_metrics['average_latency'])
            print(f"Average Packet Latency: {avg_latency:.2f}s")


class InteractiveDronePlacementGUI:
    """Interactive GUI for placing drones before simulation"""
    
    def __init__(self, area_size: Tuple[float, float] = (2000, 2000)):
        self.area_size = area_size
        self.controller = DroneSwarmACOController()
        self.placement_mode = DroneType.WORKER
        self.fig, self.ax = None, None
        self.z_height = 150.0
        
    def start_placement_interface(self):
        """Start the interactive placement interface"""
        print("\n" + "="*60)
        print("INTERACTIVE DRONE PLACEMENT INTERFACE")
        print("="*60)
        print("CONTROLS:")
        print("  - LEFT CLICK: Place drone at cursor position")
        print("  - Press 'w': Switch to WORKER drone mode")
        print("  - Press 'l': Switch to LEADER drone mode")
        print("  - Press 'c': Clear all drones")
        print("  - Press 'd': Done placing - start simulation")
        print("  - Press 'q': Quit without simulation")
        print("  - Press '+'/'-': Adjust altitude (Z-height)")
        print("="*60)
        print(f"\nArea size: {self.area_size[0]}m x {self.area_size[1]}m")
        print(f"Initial mode: {self.placement_mode.value.upper()}")
        print(f"Initial altitude: {self.z_height}m\n")
        
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        self.fig.canvas.manager.set_window_title('Drone Swarm Placement Interface')
        
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
        
        plt.tight_layout()
        plt.show()
        
    def _update_title(self):
        """Update the title with current mode and stats"""
        leader_count = sum(1 for d in self.controller.drones.values() if d.drone_type == DroneType.LEADER)
        worker_count = sum(1 for d in self.controller.drones.values() if d.drone_type == DroneType.WORKER)
        
        title = f"Drone Placement - Mode: {self.placement_mode.value.upper()} | "
        title += f"Altitude: {self.z_height:.0f}m | "
        title += f"Leaders: {leader_count} | Workers: {worker_count}"
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
    def _add_legend(self):
        """Add legend to the plot"""
        from matplotlib.lines import Line2D
        
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', label='Leader Drone',
                   markerfacecolor='red', markersize=12, alpha=0.7),
            Line2D([0], [0], marker='o', color='w', label='Worker Drone',
                   markerfacecolor='blue', markersize=10, alpha=0.7),
            Line2D([0], [0], color='gray', alpha=0.3, label='Communication Range')
        ]
        
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
    def _add_instructions(self):
        """Add instruction text to the plot"""
        instructions = "w=Worker | l=Leader | c=Clear | d=Done | q=Quit | +/- = Altitude"
        self.ax.text(0.5, -0.08, instructions, transform=self.ax.transAxes,
                    ha='center', fontsize=10, bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
        
    def _on_click(self, event):
        """Handle mouse click events"""
        if event.inaxes != self.ax:
            return
            
        if event.button == 1:
            x, y = event.xdata, event.ydata
            
            self.controller.add_drone_manual(x, y, self.z_height, self.placement_mode)
            
            self._redraw_drones()
            
    def _on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'w':
            self.placement_mode = DroneType.WORKER
            print(f"Switched to WORKER mode (altitude: {self.z_height}m)")
            self._update_title()
            self.fig.canvas.draw()
            
        elif event.key == 'l':
            self.placement_mode = DroneType.LEADER
            print(f"Switched to LEADER mode (altitude: {self.z_height}m)")
            self._update_title()
            self.fig.canvas.draw()
            
        elif event.key == 'c':
            self.controller.drones.clear()
            print("All drones cleared!")
            self._redraw_drones()
            
        elif event.key == 'd':
            if len(self.controller.drones) == 0:
                print("ERROR: No drones placed! Please place at least one drone.")
                return
            print(f"\nPlacement complete! Total drones: {len(self.controller.drones)}")
            plt.close(self.fig)
            self._start_simulation()
            
        elif event.key == 'q':
            print("\nExiting without simulation...")
            plt.close(self.fig)
            
        elif event.key == '+' or event.key == '=':
            self.z_height = min(500, self.z_height + 20)
            print(f"Altitude increased to: {self.z_height}m")
            self._update_title()
            self.fig.canvas.draw()
            
        elif event.key == '-' or event.key == '_':
            self.z_height = max(10, self.z_height - 20)
            print(f"Altitude decreased to: {self.z_height}m")
            self._update_title()
            self.fig.canvas.draw()
            
    def _redraw_drones(self):
        """Redraw all drones and their communication ranges"""
        self.ax.clear()
        
        self.ax.set_xlim(0, self.area_size[0])
        self.ax.set_ylim(0, self.area_size[1])
        self.ax.set_xlabel('X Position (meters)', fontsize=12)
        self.ax.set_ylabel('Y Position (meters)', fontsize=12)
        self.ax.grid(True, alpha=0.3, linestyle='--')
        self.ax.set_aspect('equal')
        
        for drone in self.controller.drones.values():
            color = 'red' if drone.drone_type == DroneType.LEADER else 'blue'
            circle = Circle((drone.position.x, drone.position.y), 
                          drone.communication_range, 
                          color=color, alpha=0.05, linestyle='--', 
                          fill=True, linewidth=1)
            self.ax.add_patch(circle)
        
        for drone in self.controller.drones.values():
            if drone.drone_type == DroneType.LEADER:
                color = 'red'
                marker = 's'
                size = 150
            else:
                color = 'blue'
                marker = 'o'
                size = 100
                
            self.ax.scatter(drone.position.x, drone.position.y, 
                          c=color, marker=marker, s=size, 
                          alpha=0.7, edgecolors='black', linewidth=1.5)
            
            self.ax.text(drone.position.x, drone.position.y + 50, 
                        drone.drone_id, fontsize=8, ha='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        self._update_title()
        self._add_legend()
        self._add_instructions()
        
        self.fig.canvas.draw()
        
    def _start_simulation(self):
        """Start the simulation with placed drones"""
        print("\n" + "="*60)
        print("INITIALIZING SIMULATION")
        print("="*60)
        
        self.controller.update_neighbor_relationships()
        
        print("\nShowing initial network configuration...")
        self.controller.visualize_network()
        
        print("\nWhat would you like to do?")
        print("1. Run full simulation")
        print("2. Test routing between specific drones")
        print("3. Test routing with random drones")
        
        try:
            choice = input("Enter choice (1-3, default 1): ").strip() or "1"
        except:
            choice = "1"
        
        if choice == "2":
            print("\nAvailable drones:")
            for drone_id in self.controller.drones.keys():
                print(f"  - {drone_id}")
            
            try:
                source = input("Enter source drone ID: ").strip()
                dest = input("Enter destination drone ID: ").strip()
                self.controller.test_routing_between_drones(source, dest)
            except:
                print("Invalid input, using random drones...")
                self.controller.test_routing_between_drones()
            
            try:
                run_sim = input("\nRun full simulation? (y/n, default n): ").strip().lower()
            except:
                run_sim = "n"
            
            if run_sim != "y":
                return
        
        elif choice == "3":
            self.controller.test_routing_between_drones()
            
            try:
                run_sim = input("\nRun full simulation? (y/n, default n): ").strip().lower()
            except:
                run_sim = "n"
            
            if run_sim != "y":
                return
        
        print("\nHow long should the simulation run?")
        try:
            duration = int(input("Enter duration in seconds (default 60): ") or "60")
        except ValueError:
            duration = 60
        
        self.controller.run_simulation(duration=duration)
        
        print("\nShowing final network state...")
        self.controller.visualize_network()
        
        try:
            test_after = input("\nTest routing after simulation? (y/n, default n): ").strip().lower()
            if test_after == "y":
                self.controller.test_routing_between_drones()
        except:
            pass


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DRONE SWARM ACO ROUTING SIMULATION")
    print("  Interactive Placement Interface")
    print("="*60)
    
    print("\nEnter simulation area dimensions:")
    try:
        width = float(input("Width (meters, default 2000): ") or "2000")
        height = float(input("Height (meters, default 2000): ") or "2000")
    except ValueError:
        width, height = 2000, 2000
        print("Using default area size: 2000m x 2000m")
    
    gui = InteractiveDronePlacementGUI(area_size=(width, height))
    gui.start_placement_interface()
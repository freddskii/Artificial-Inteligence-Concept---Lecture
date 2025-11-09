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
    rssi: float  # Received Signal Strength Indicator
    latency: float  # in milliseconds
    packet_loss: float  # 0.0 to 1.0
    bandwidth: float  # in Mbps
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
        self.ttl = 30.0  # Time to live in seconds
        
    def record_hop(self, drone_id: str, link_quality: float, position: Position):
        """Record a hop in the path with quality metrics"""
        self.path_taken.append({
            'drone_id': drone_id,
            'position': position,
            'link_quality': link_quality,
            'timestamp': time.time()
        })
        self.hop_count += 1
        
    def should_continue(self) -> bool:
        """Check if ant should continue its journey"""
        if self.hop_count >= self.max_hops:
            return False
        if time.time() - self.timestamp > self.ttl:
            return False
        if self.path_taken and self.path_taken[-1]['drone_id'] == self.destination_drone.drone_id:
            return False
        return True
        
    def calculate_current_path_quality(self) -> float:
        """Calculate overall path quality based on recorded metrics"""
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
        """Calculate comprehensive path quality metrics"""
        if not self.forward_ant.path_taken:
            return {}
            
        total_latency = 0.0
        total_quality = 0.0
        min_energy = float('inf')
        
        for hop in self.forward_ant.path_taken:
            total_quality += hop['link_quality']
            
        avg_quality = total_quality / len(self.forward_ant.path_taken)
        hop_penalty = len(self.forward_ant.path_taken) * 0.1
        
        # Composite score (higher is better)
        self.path_quality_score = avg_quality - hop_penalty
        return {
            'quality_score': self.path_quality_score,
            'hop_count': len(self.forward_ant.path_taken),
            'avg_quality': avg_quality
        }
        
    def update_pheromones(self, drone_network: Dict[str, 'IntelligentDrone']):
        """Update pheromones along the path"""
        metrics = self.calculate_path_metrics()
        path_quality = max(0.1, metrics['quality_score'])  # Ensure positive
        
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
        self.communication_range = 2000.0 if drone_type == DroneType.LEADER else 1000.0  # meters
        self.max_speed = 50.0  # km/h
        self.current_mission = None
        
        # ACO components
        self.pheromone_table = {}  # {destination_drone_id: {neighbor_id: pheromone_value}}
        self.neighbor_drones = {}  # {neighbor_id: LinkMetrics}
        self.routing_table = {}    # {destination: next_hop}
        
        # Queues
        self.ant_queue = deque()
        self.data_queue = deque()
        
        # Performance tracking
        self.packets_forwarded = 0
        self.ants_processed = 0
        self.energy_used = 0.0
        
        # Thread safety
        self.lock = threading.RLock()
        
    def update_position(self, new_position: Position, time_delta: float = 1.0):
        """Update drone position and handle movement energy consumption"""
        distance_moved = self.position.distance_to(new_position)
        movement_energy = distance_moved * 0.01  # Energy per meter
        
        with self.lock:
            self.position = new_position
            self.battery_level -= movement_energy
            self.energy_used += movement_energy
            
    def measure_link_quality(self, neighbor_drone: 'IntelligentDrone') -> float:
        """Measure link quality to neighbor drone (0.0 to 1.0)"""
        distance = self.position.distance_to(neighbor_drone.position)
        
        if distance > self.communication_range:
            return 0.0
            
        # Calculate line of sight (simplified)
        los_quality = 1.0 - (distance / self.communication_range) ** 2
        
        # Signal quality based on distance and drone type
        if self.drone_type == DroneType.LEADER or neighbor_drone.drone_type == DroneType.LEADER:
            signal_boost = 1.2
        else:
            signal_boost = 1.0
            
        # Energy factor
        energy_factor = min(self.battery_level, neighbor_drone.battery_level) / 100.0
        
        link_quality = los_quality * signal_boost * energy_factor
        return max(0.0, min(1.0, link_quality))
        
    def calculate_heuristic(self, neighbor_drone: 'IntelligentDrone', destination: 'IntelligentDrone') -> float:
        """Calculate heuristic value η(i,j) for ACO"""
        # Line of sight quality
        los_quality = self.measure_link_quality(neighbor_drone)
        
        # Signal strength (distance-based)
        distance_to_neighbor = self.position.distance_to(neighbor_drone.position)
        signal_strength = 1.0 - (distance_to_neighbor / self.communication_range)
        
        # Battery compatibility
        battery_compatibility = min(self.battery_level, neighbor_drone.battery_level) / 100.0
        
        # Mobility synchronization (predict link duration)
        mobility_sync = self.predict_link_duration(neighbor_drone)
        
        # Progress toward destination
        current_to_dest = self.position.distance_to(destination.position)
        neighbor_to_dest = neighbor_drone.position.distance_to(destination.position)
        progress = 1.0 if neighbor_to_dest < current_to_dest else 0.5
        
        # Weighted combination
        heuristic = (0.30 * los_quality + 
                    0.25 * signal_strength + 
                    0.20 * battery_compatibility + 
                    0.15 * mobility_sync + 
                    0.10 * progress)
        
        return max(0.1, heuristic)  # Ensure minimum heuristic
        
    def predict_link_duration(self, neighbor_drone: 'IntelligentDrone') -> float:
        """Predict how long the link will remain stable"""
        # Simplified prediction based on relative positions and speeds
        return 0.8  # Placeholder - would use real mobility prediction
        
    def process_forward_ant(self, ant: ForwardAnt) -> Optional[str]:
        """Process forward ant and decide next hop"""
        with self.lock:
            self.ants_processed += 1
            
            # Record this hop
            link_quality = 1.0  # Default, would measure from neighbor info
            ant.record_hop(self.drone_id, link_quality, self.position)
            
            # Check if we reached destination
            if self.drone_id == ant.destination_drone.drone_id:
                return None  # Ant reached destination
                
            # Check termination conditions
            if not ant.should_continue():
                return None
                
            # Get available neighbors (excluding previous hops)
            available_neighbors = self.get_available_neighbors(ant)
            if not available_neighbors:
                return None
                
            # Select next hop using ACO probability
            next_hop_id = self.probabilistic_routing_decision(available_neighbors, ant.destination_drone)
            return next_hop_id
            
    def get_available_neighbors(self, ant: ForwardAnt) -> List['IntelligentDrone']:
        """Get neighbors that can help reach destination"""
        available = []
        visited_drones = {hop['drone_id'] for hop in ant.path_taken}
        
        for neighbor_id, metrics in self.neighbor_drones.items():
            if neighbor_id not in visited_drones:
                # In real implementation, we'd have reference to neighbor drone objects
                # For now, we'll return neighbor IDs and let controller handle
                available.append(neighbor_id)
                
        return available
        
    def probabilistic_routing_decision(self, available_neighbors: List[str], destination: 'IntelligentDrone') -> str:
        """Select next hop using ACO probability formula"""
        if not available_neighbors:
            return random.choice(available_neighbors) if available_neighbors else None
            
        probabilities = []
        total = 0.0
        
        alpha = 0.6  # Pheromone importance
        beta = 0.4   # Heuristic importance
        
        for neighbor_id in available_neighbors:
            # Get pheromone value (initialize if not exists)
            if destination.drone_id not in self.pheromone_table:
                self.pheromone_table[destination.drone_id] = {}
            if neighbor_id not in self.pheromone_table[destination.drone_id]:
                self.pheromone_table[destination.drone_id][neighbor_id] = 0.1
                
            pheromone = self.pheromone_table[destination.drone_id][neighbor_id]
            
            # Get heuristic (simplified - would use actual drone reference)
            heuristic = 0.7  # Placeholder
            
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
                    
        # Fallback: random selection
        return random.choice(available_neighbors)
        
    def process_backward_ant(self, ant: BackwardAnt):
        """Process backward ant and update routing information"""
        with self.lock:
            # Update routing table based on path quality
            if ant.path_quality_score > 0.5:  # Good quality threshold
                # Extract next hop from the path
                if ant.forward_ant.path_taken:
                    first_hop = ant.forward_ant.path_taken[0]
                    if len(ant.forward_ant.path_taken) > 1:
                        next_hop = ant.forward_ant.path_taken[1]['drone_id']
                        self.routing_table[ant.forward_ant.destination_drone.drone_id] = next_hop

    def update_pheromone(self, neighbor_id: str, path_quality: float):
        """Update pheromone value for a neighbor"""
        evaporation_rate = 0.3
        Q = 2.0  # Reinforcement constant
        
        with self.lock:
            # Initialize if not exists
            if neighbor_id not in self.pheromone_table:
                self.pheromone_table[neighbor_id] = {}
            
            # Get current pheromone or initialize
            current_pheromone = self.pheromone_table.get(neighbor_id, {}).get('default', 0.1)
            
            # Evaporation and reinforcement
            evaporated_pheromone = current_pheromone * (1 - evaporation_rate)
            reinforcement = Q * path_quality
            new_pheromone = evaporated_pheromone + reinforcement
            
            # Bound checking
            new_pheromone = max(0.1, min(1.0, new_pheromone))
            
            # Update pheromone table
            if 'default' not in self.pheromone_table[neighbor_id]:
                self.pheromone_table[neighbor_id]['default'] = new_pheromone
            else:
                self.pheromone_table[neighbor_id]['default'] = new_pheromone

    def route_data_packet(self, packet: Dict) -> Optional[str]:
        """Route data packet using pheromone-based routing table"""
        with self.lock:
            destination = packet.get('destination')
            if destination in self.routing_table:
                self.packets_forwarded += 1
                transmission_energy = 0.05  # Energy per packet
                self.battery_level -= transmission_energy
                self.energy_used += transmission_energy
                return self.routing_table[destination]
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

    def get_status(self) -> Dict:
        """Get current drone status"""
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
        
        # ACO parameters
        self.aco_params = {
            'alpha': 0.6,  # Pheromone importance
            'beta': 0.4,   # Heuristic importance
            'rho': 0.3,    # Evaporation rate
            'Q': 2.0,      # Reinforcement constant
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
        
        # Threading
        self.ant_thread = None
        self.monitor_thread = None
        self.lock = threading.RLock()

    def initialize_swarm(self, num_drones: int, area_size: Tuple[float, float, float] = (5000, 5000, 500)):
        """Initialize drone swarm in the specified area"""
        print(f"Initializing swarm with {num_drones} drones...")
        
        # Create leader drones (10% of total)
        num_leaders = max(1, num_drones // 10)
        for i in range(num_leaders):
            position = Position(
                x=random.uniform(0, area_size[0]),
                y=random.uniform(0, area_size[1]),
                z=random.uniform(100, area_size[2])
            )
            drone_id = f"leader_{i}"
            self.drones[drone_id] = IntelligentDrone(drone_id, position, DroneType.LEADER, 150.0)
        
        # Create worker drones
        for i in range(num_drones - num_leaders):
            position = Position(
                x=random.uniform(0, area_size[0]),
                y=random.uniform(0, area_size[1]),
                z=random.uniform(50, area_size[2])
            )
            drone_id = f"worker_{i}"
            self.drones[drone_id] = IntelligentDrone(drone_id, position, DroneType.WORKER, 100.0)
        
        # Initialize neighbor relationships
        self.update_neighbor_relationships()
        print("Swarm initialization completed!")

    def update_neighbor_relationships(self):
        """Update neighbor relationships based on current positions"""
        drone_ids = list(self.drones.keys())
        
        for i, drone_id1 in enumerate(drone_ids):
            drone1 = self.drones[drone_id1]
            
            # Clear existing neighbors
            drone1.neighbor_drones.clear()
            
            for drone_id2 in drone_ids[i+1:]:
                drone2 = self.drones[drone_id2]
                distance = drone1.position.distance_to(drone2.position)
                
                # Check if drones are within communication range
                if distance <= drone1.communication_range:
                    # Create link metrics
                    link_quality = drone1.measure_link_quality(drone2)
                    if link_quality > 0.1:  # Minimum quality threshold
                        link_metrics = LinkMetrics(
                            rssi=link_quality * 100,
                            latency=distance * 0.01,  # Simplified latency model
                            packet_loss=1.0 - link_quality,
                            bandwidth=10.0 * link_quality,
                            last_updated=time.time()
                        )
                        drone1.add_neighbor(drone_id2, link_metrics)
                        drone2.add_neighbor(drone_id1, link_metrics)

    def launch_ant_agents(self):
        """Launch ant agents for route discovery"""
        if not self.drones:
            return
            
        drone_ids = list(self.drones.keys())
        
        # Launch ants from random sources to random destinations
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
        """Process forward ant through the network"""
        current_drone = ant.source_drone
        start_time = time.time()
        max_processing_time = 10.0  # seconds
        
        while current_drone and ant.should_continue() and (time.time() - start_time) < max_processing_time:
            next_hop_id = current_drone.process_forward_ant(ant)
            
            if not next_hop_id:
                break
                
            if next_hop_id in self.drones:
                current_drone = self.drones[next_hop_id]
            else:
                break
        
        # If ant reached destination or completed journey, create backward ant
        if ant.path_taken and ant.path_taken[-1]['drone_id'] == ant.destination_drone.drone_id:
            backward_ant = BackwardAnt(ant)
            self.process_backward_ant(backward_ant)

    def process_backward_ant(self, ant: BackwardAnt):
        """Process backward ant through the return path"""
        ant.calculate_path_metrics()
        ant.update_pheromones(self.drones)
        
        # Update performance metrics
        route_discovery_time = time.time() - ant.forward_ant.timestamp
        self.performance_metrics['route_discovery_time'].append(route_discovery_time)

    def update_drone_positions(self):
        """Update drone positions based on mission type"""
        for drone in self.drones.values():

            # makes a random position between -50 and 50 relative to a drone
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
            else:  # SEARCH_RESCUE
                # Systematic search pattern
                new_position = Position(
                    x=drone.position.x + random.uniform(-30, 30),
                    y=drone.position.y + random.uniform(-30, 30),
                    z=drone.position.z + random.uniform(-5, 5)
                )
            
            # Ensure positions stay within bounds
            new_position.x = max(0, min(5000, new_position.x))
            new_position.y = max(0, min(5000, new_position.y))
            new_position.z = max(10, min(500, new_position.z))
            
            drone.update_position(new_position)

    def simulate_data_traffic(self):
        """Simulate data traffic between drones"""
        if len(self.drones) < 2:
            return
            
        drone_ids = list(self.drones.keys())
        
        # Generate random data packets
        num_packets = random.randint(1, len(drone_ids) // 4)
        
        for _ in range(num_packets):
            source_id = random.choice(drone_ids)
            destination_id = random.choice([did for did in drone_ids if did != source_id])
            
            packet = {
                'source': source_id,
                'destination': destination_id,
                'size': random.randint(100, 5000),  # bytes
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
        max_hops = 15 # max hop
        hop_count = 0
        
        while hop_count < max_hops:
            next_hop_id = current_drone.route_data_packet(packet)
            
            if not next_hop_id or next_hop_id not in self.drones:
                break
                
            current_drone = self.drones[next_hop_id]
            path.append(next_hop_id)
            hop_count += 1
            
            if next_hop_id == destination_id:
                # Packet delivered successfully
                latency = time.time() - packet['timestamp']
                self.performance_metrics['average_latency'].append(latency)
                break

    def adaptive_parameter_tuning(self):
        """Dynamically adjust ACO parameters based on network conditions"""
        total_drones = len(self.drones)
        if total_drones == 0:
            return
            
        # Calculate network density
        avg_neighbors = sum(len(drone.neighbor_drones) for drone in self.drones.values()) / total_drones
        network_density = avg_neighbors / total_drones
        
        # Adjust parameters based on density
        if network_density > 0.3:  # High density
            self.aco_params['alpha'] = 0.7  # Favor pheromone (exploitation)
            self.aco_params['beta'] = 0.3   # Reduce heuristic
            self.aco_params['rho'] = 0.4    # Faster evaporation
        elif network_density < 0.1:  # Low density
            self.aco_params['alpha'] = 0.4  # Reduce pheromone
            self.aco_params['beta'] = 0.6   # Favor heuristic (exploration)
            self.aco_params['rho'] = 0.2    # Slower evaporation
        else:  # Medium density
            self.aco_params['alpha'] = 0.6
            self.aco_params['beta'] = 0.4
            self.aco_params['rho'] = 0.3

    def handle_emergency(self, emergency_drone_id: str, emergency_type: str):
        """Handle emergency situations"""
        print(f"EMERGENCY: {emergency_type} for drone {emergency_drone_id}")
        self.emergency_mode = True
        
        if emergency_type == "BATTERY_CRITICAL":
            # Find nearest charging station or return to base
            emergency_drone = self.drones.get(emergency_drone_id)
            if emergency_drone:
                # Force high-priority routing to base
                base_drone = self.find_base_station()
                if base_drone:
                    self.force_emergency_route(emergency_drone, base_drone)
        
        elif emergency_type == "CONNECTION_LOST":
            # Attempt to re-establish connection
            self.reconnect_drone(emergency_drone_id)

    def find_base_station(self) -> Optional[IntelligentDrone]:
        """Find a base station or leader drone"""
        for drone in self.drones.values():
            if drone.drone_type == DroneType.LEADER:
                return drone
        return None

    def force_emergency_route(self, source_drone: IntelligentDrone, destination_drone: IntelligentDrone):
        """Force establish emergency route"""
        # Create emergency ant with highest priority
        ant_id = f"emergency_ant_{self.ant_counter}"
        self.ant_counter += 1
        
        emergency_ant = ForwardAnt(ant_id, source_drone, destination_drone, AntType.EMERGENCY)
        emergency_ant.max_hops = 10  # Limit hops for emergency
        
        self.process_forward_ant(emergency_ant)

    def reconnect_drone(self, drone_id: str):
        """Attempt to reconnect a disconnected drone"""
        drone = self.drones.get(drone_id)
        if not drone:
            return
            
        # Reset pheromone table to encourage exploration
        drone.pheromone_table.clear()
        
        # Launch exploratory ants
        for _ in range(3):  # Launch multiple exploratory ants
            other_drones = [d for d in self.drones.values() if d.drone_id != drone_id]
            if other_drones:
                destination = random.choice(other_drones)
                ant_id = f"explore_ant_{self.ant_counter}"
                self.ant_counter += 1
                
                explore_ant = ForwardAnt(ant_id, drone, destination, AntType.EXPLORATORY)
                self.process_forward_ant(explore_ant)

    def monitor_network_health(self):
        """Monitor overall network health and performance"""
        total_drones = len(self.drones)
        if total_drones == 0:
            return
            
        # Calculate metrics
        avg_battery = sum(drone.battery_level for drone in self.drones.values()) / total_drones
        total_packets = sum(drone.packets_forwarded for drone in self.drones.values())
        total_ants = sum(drone.ants_processed for drone in self.drones.values())
        
        # Check for emergencies
        low_battery_drones = [drone_id for drone_id, drone in self.drones.items() 
                             if drone.battery_level < 20.0]
        
        for drone_id in low_battery_drones:
            self.handle_emergency(drone_id, "BATTERY_CRITICAL")
        
        # Print status periodically
        if self.simulation_time % 10 == 0:
            print(f"\n--- Network Status at Time {self.simulation_time} ---")
            print(f"Active Drones: {total_drones}")
            print(f"Average Battery: {avg_battery:.1f}%")
            print(f"Total Packets Forwarded: {total_packets}")
            print(f"Total Ants Processed: {total_ants}")
            print(f"Emergency Mode: {self.emergency_mode}")

    def visualize_network(self):
        """Create a simple visualization of the drone network"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot drones
            for drone_id, drone in self.drones.items():
                color = 'red' if drone.drone_type == DroneType.LEADER else 'blue'
                marker = 's' if drone.drone_type == DroneType.LEADER else 'o'
                size = 100 if drone.drone_type == DroneType.LEADER else 50
                
                plt.scatter(drone.position.x, drone.position.y, c=color, 
                          marker=marker, s=size, label=drone_id if drone.drone_type == DroneType.LEADER else "")
                
                # Plot connections
                for neighbor_id in drone.neighbor_drones.keys():
                    if neighbor_id in self.drones:
                        neighbor = self.drones[neighbor_id]
                        plt.plot([drone.position.x, neighbor.position.x],
                                [drone.position.y, neighbor.position.y], 'gray', alpha=0.3)
            
            plt.title(f"Drone Swarm Network - Time: {self.simulation_time}")
            plt.xlabel("X Position (m)")
            plt.ylabel("Y Position (m)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Visualization error: {e}")

    def run_simulation(self, duration: int = 300):
        """Main simulation loop"""
        print("Starting Drone Swarm ACO Simulation...")
        self.running = True
        start_time = time.time()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        # Start ant generation thread
        self.ant_thread = threading.Thread(target=self._ant_generation_loop, daemon=True)
        self.ant_thread.start()
        
        try:
            while self.running and (time.time() - start_time) < duration:
                loop_start = time.time()
                
                # Update simulation state
                self.simulation_time += 1
                
                # Update drone positions
                self.update_drone_positions()
                
                # Update neighbor relationships
                self.update_neighbor_relationships()
                
                # Simulate data traffic
                self.simulate_data_traffic()
                
                # Adaptive parameter tuning
                self.adaptive_parameter_tuning()
                
                # Monitor network health
                self.monitor_network_health()
                
                # Control simulation speed
                loop_time = time.time() - loop_start
                sleep_time = max(0, 1.0 - loop_time)  # Aim for 1 second per iteration
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
        finally:
            self.running = False
            self._print_final_stats()

    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.running:
            self.monitor_network_health()
            time.sleep(5)  # Check every 5 seconds

    def _ant_generation_loop(self):
        """Background ant generation loop"""
        while self.running:
            self.launch_ant_agents()
            time.sleep(self.aco_params['ant_generation_interval'])

    def _print_final_stats(self):
        """Print final simulation statistics"""
        print("\n" + "="*50)
        print("FINAL SIMULATION STATISTICS")
        print("="*50)
        
        total_drones = len(self.drones)
        if total_drones == 0:
            return
            
        # Calculate final metrics
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
        
        # Performance metrics
        if self.performance_metrics['route_discovery_time']:
            avg_discovery_time = np.mean(self.performance_metrics['route_discovery_time'])
            print(f"Average Route Discovery Time: {avg_discovery_time:.2f}s")
        
        if self.performance_metrics['average_latency']:
            avg_latency = np.mean(self.performance_metrics['average_latency'])
            print(f"Average Packet Latency: {avg_latency:.2f}s")

# Example usage and testing
if __name__ == "__main__":
    # Create and run the simulation
    controller = DroneSwarmACOController()
    
    # Initialize with 20 drones
    controller.initialize_swarm(25, area_size=(2000, 2000, 300))
    
    # Run simulation for 2 minutes
    print("Starting 2-minute simulation...")
    controller.run_simulation(duration=60)
    
    # Optional: Visualize final network state
    print("\nGenerating network visualization...")
    controller.visualize_network()

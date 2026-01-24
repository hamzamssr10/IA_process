"""UDP communication module for sending detection data."""
import socket
import time
from threading import Lock
from typing import Set, Tuple

# Global variables
sock = None
dest_port = None
connected_clients: Set[Tuple[str, int]] = set()
clients_lock = Lock()
stop_listener_flag = False


def calculate_checksum(data):
    """Calculate checksum for UDP packet."""
    checksum = sum(data[1:]) & 0xFF
    return checksum


def start_udp(port=52383):
    """Start UDP socket."""
    global sock, dest_port

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        sock.settimeout(1.0)
        dest_port = port
        return True
    except Exception as e:
        print(f"❌ UDP start error: {e}")
        return False


def stop_udp():
    """Close UDP socket."""
    global sock, stop_listener_flag

    stop_listener_flag = True

    if sock:
        sock.close()
        sock = None
        print("✓ UDP Socket closed")


def listen_for_clients_thread():
    """Listen for client messages and register them."""
    global sock, connected_clients, stop_listener_flag

    print("🎧 Client listener thread started...")

    while not stop_listener_flag:
        try:
            try:
                data, addr = sock.recvfrom(1024)

                with clients_lock:
                    if addr not in connected_clients:
                        connected_clients.add(addr)

                    try:
                        message = data.decode("utf-8", errors="ignore").strip()
                        if message == "DISCONNECT":
                            connected_clients.discard(addr)
                    except:
                        pass

            except socket.timeout:
                continue

        except Exception as e:
            if not stop_listener_flag:
                print(f"⚠ Listen error: {e}")
            time.sleep(0.1)

    print("🛑 Client listener thread stopped")


def convert_and_send(json_data):
    """Convert JSON data and send via UDP to all connected clients."""
    global sock, connected_clients

    if not sock:
        print("⚠ UDP Socket not started.")
        return False

    with clients_lock:
        if not connected_clients:
            return False

        try:
            data = bytearray()
            data.append(0xFB)  # Header
            
            nb_objects = json_data["len"]
            data.append(nb_objects)

            for obj in json_data["data"]:
                cls = int(obj["CLS"], 16) if isinstance(obj["CLS"], str) else obj["CLS"]
                id_track = int(obj["ID_TRACK"], 16) if isinstance(obj["ID_TRACK"], str) else obj["ID_TRACK"]

                x1_hex = obj["X1"] & 0xFFFF
                x2_hex = obj["X2"] & 0xFFFF
                y1_hex = obj["Y1"] & 0xFFFF
                y2_hex = obj["Y2"] & 0xFFFF
                z_hex = obj.get("Z", 0) & 0xFFFFFFFF

                data.append(cls & 0xFF)
                data.append(id_track & 0xFF)
                
                # X1 (2 bytes - big endian)
                data.append((x1_hex >> 8) & 0xFF)
                data.append(x1_hex & 0xFF)
                
                # X2
                data.append((x2_hex >> 8) & 0xFF)
                data.append(x2_hex & 0xFF)
                
                # Y1
                data.append((y1_hex >> 8) & 0xFF)
                data.append(y1_hex & 0xFF)
                
                # Y2
                data.append((y2_hex >> 8) & 0xFF)
                data.append(y2_hex & 0xFF)
                
                # Z (4 bytes)
                data.append((z_hex >> 24) & 0xFF)
                data.append((z_hex >> 16) & 0xFF)
                data.append((z_hex >> 8) & 0xFF)
                data.append(z_hex & 0xFF)

            checksum = calculate_checksum(data)
            data.append(checksum & 0xFF)

            success_count = 0
            failed_clients = []

            for client_addr in list(connected_clients):
                try:
                    sock.sendto(data, client_addr)
                    success_count += 1
                except Exception as e:
                    failed_clients.append(client_addr)

            for failed in failed_clients:
                connected_clients.discard(failed)

            return True

        except Exception as e:
            print(f"❌ Send error: {e}")
            return False

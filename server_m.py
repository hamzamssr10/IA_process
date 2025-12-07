import socket
import json
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Set, Tuple
import uvicorn
from datetime import datetime

# Variables globales
sock = None
dest_port = 5012
sending_task = None
listener_task = None

# Liste des clients connectés (IP, Port)
connected_clients: Set[Tuple[str, int]] = set()

# Données statiques pour test
STATIC_DATA = {
    "len": 3,
    "data": [
        {"CLS": "0x0", "ID_TRACK": "0x1a", "X": 150, "Y": 90, "Z": 50},
        {"CLS": "0x2", "ID_TRACK": "0x3f", "X": 410, "Y": 210, "Z": 100},
        {"CLS": "0x7", "ID_TRACK": "0x12", "X": 75, "Y": 120, "Z": 25},
    ],
}


# Modèle Pydantic pour validation
class ObjectData(BaseModel):
    CLS: str
    ID_TRACK: str
    X: int
    Y: int
    Z: Optional[int] = 0


class DetectionData(BaseModel):
    len: int
    data: List[ObjectData]


# Création de l'application FastAPI
app = FastAPI(title="UDP Server API - Simple Checksum")


def calculate_checksum(data):
    """
    Calcule le checksum simple (somme des données seulement, sans header ni checksum)
    Checksum = (Byte2 + Byte3 + Byte4 + ... + ByteN) & 0xFF

    Args:
        data: bytearray des données (incluant le header, sans le checksum final)

    Returns:
        int: Checksum (1 byte)
    """
    # Somme de tous les bytes sauf le premier (header 0xFB)
    # Le checksum ne s'inclut pas lui-même dans le calcul
    checksum = sum(data[1:]) & 0xFF
    return checksum


def start_udp(port=dest_port):
    """Démarre le socket UDP"""
    global sock, dest_port

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
        sock.setblocking(False)  # Mode non-bloquant pour asyncio
        dest_port = port
        print(f" UDP Socket démarré sur 0.0.0.0:{dest_port}")
        return True
    except Exception as e:
        print(f" Erreur démarrage UDP: {e}")
        return False


async def listen_for_clients():
    """Écoute les messages des clients pour les enregistrer"""
    global sock, connected_clients

    print(" Écoute des clients sur le port UDP...")

    while True:
        try:
            # Utiliser asyncio pour recevoir des données sans bloquer
            loop = asyncio.get_event_loop()
            data, addr = await loop.sock_recvfrom(sock, 1024)

            # Enregistrer le client s'il n'est pas déjà dans la liste
            if addr not in connected_clients:
                connected_clients.add(addr)
                print(f" Nouveau client connecté: {addr[0]}:{addr[1]}")
                print(f" Total clients: {len(connected_clients)}")

            # Si le client envoie "DISCONNECT", le retirer
            if data.decode("utf-8", errors="ignore").strip() == "DISCONNECT":
                connected_clients.discard(addr)
                print(f" Client déconnecté: {addr[0]}:{addr[1]}")
                print(f" Total clients: {len(connected_clients)}")

        except asyncio.CancelledError:
            print("\n Tâche d'écoute arrêtée")
            break
        except Exception as e:
            # Ignorer les erreurs de socket non-bloquant
            if "Resource temporarily unavailable" not in str(e):
                print(f" Erreur écoute: {e}")
            await asyncio.sleep(0.1)


def convert_and_send(json_data):
    """
    Convertit les données JSON et envoie via UDP à tous les clients connectés

    Args:
        json_data: Dictionnaire avec 'len' et 'data'

    Returns:
        bool: True si envoi réussi, False sinon
    """
    global sock, connected_clients

    if not sock:
        print(" Socket UDP non démarré.")
        return False

    if not connected_clients:
        print(" Aucun client connecté - aucun envoi effectué")
        return False

    try:
        # Construction du paquet
        data = bytearray()

        # Header
        data.append(0xFB)

        # Nombre d'objets
        nb_objects = json_data["len"]
        data.append(nb_objects)

        # Boucle sur chaque objet
        for obj in json_data["data"]:
            # Conversion des valeurs hexadécimales string en int
            cls = int(obj["CLS"], 16) if isinstance(obj["CLS"], str) else obj["CLS"]
            id_track = (
                int(obj["ID_TRACK"], 16)
                if isinstance(obj["ID_TRACK"], str)
                else obj["ID_TRACK"]
            )

            # Conversion X, Y et Z en hexadécimal (4 bytes chacun)
            x_hex = obj["X"] & 0xFFFFFFFF
            y_hex = obj["Y"] & 0xFFFFFFFF
            z_hex = obj.get("Z", 0) & 0xFFFFFFFF

            # Ajout au paquet
            data.append(cls & 0xFF)
            data.append(id_track & 0xFF)

            # X (4 bytes - big endian)
            data.append((x_hex >> 24) & 0xFF)
            data.append((x_hex >> 16) & 0xFF)
            data.append((x_hex >> 8) & 0xFF)
            data.append(x_hex & 0xFF)

            # Y (4 bytes - big endian)
            data.append((y_hex >> 24) & 0xFF)
            data.append((y_hex >> 16) & 0xFF)
            data.append((y_hex >> 8) & 0xFF)
            data.append(y_hex & 0xFF)

            # Z (4 bytes - big endian)
            data.append((z_hex >> 24) & 0xFF)
            data.append((z_hex >> 16) & 0xFF)
            data.append((z_hex >> 8) & 0xFF)
            data.append(z_hex & 0xFF)

            print(
                f"  • Objet: CLS=0x{cls:02X}, ID=0x{id_track:02X}, X=0x{x_hex:08X}({obj['X']}), Y=0x{y_hex:08X}({obj['Y']}), Z=0x{z_hex:08X}({obj.get('Z', 0)})"
            )

        # Calcul du Checksum (somme de tous les bytes sauf header, avant d'ajouter le checksum)
        checksum = calculate_checksum(data)

        # Ajout du Checksum à la fin (1 byte)
        data.append(checksum & 0xFF)

        print(f"\n    Checksum: 0x{checksum:02X} (somme: NB_OBJ + données objets)")

        # Envoi à tous les clients connectés
        success_count = 0
        failed_clients = []

        for client_addr in list(connected_clients):
            try:
                sock.sendto(data, client_addr)
                success_count += 1
            except Exception as e:
                print(f" Erreur envoi vers {client_addr[0]}:{client_addr[1]}: {e}")
                failed_clients.append(client_addr)

        # Retirer les clients qui ont échoué
        for failed in failed_clients:
            connected_clients.discard(failed)
            print(f" Client retiré (échec envoi): {failed[0]}:{failed[1]}")

        print(f"\n Envoyé {len(data)} bytes à {success_count} client(s)")
        print(f"    Données: {' '.join(f'{b:02X}' for b in data[:20])}...")
        return True

    except Exception as e:
        print(f" Erreur envoi: {e}")
        return False


def stop_udp():
    """Ferme le socket UDP"""
    global sock
    if sock:
        sock.close()
        sock = None
        print(" UDP Socket fermé")


async def auto_send_task():
    """Tâche asynchrone qui envoie les données toutes les secondes"""
    counter = 0
    while True:
        try:
            counter += 1
            print(f"\n{'='*60}")
            print(
                f" Envoi automatique #{counter} - {datetime.now().strftime('%H:%M:%S')}"
            )
            print(f"    Clients connectés: {len(connected_clients)}")
            print(f"{'='*60}")

            convert_and_send(STATIC_DATA)

            print(f"\n Prochaine envoi dans 1 seconde...")
            await asyncio.sleep(1)

        except asyncio.CancelledError:
            print("\n Tâche d'envoi automatique arrêtée")
            break
        except Exception as e:
            print(f"\n Erreur dans auto_send_task: {e}")
            await asyncio.sleep(1)


@app.on_event("startup")
async def startup_event():
    """Démarre le socket UDP et les tâches au démarrage de l'API"""
    global sending_task, listener_task

    start_udp(port=dest_port)

    # Démarrer la tâche d'écoute des clients
    listener_task = asyncio.create_task(listen_for_clients())
    print(" Tâche d'écoute des clients démarrée")

    # Démarrer la tâche d'envoi automatique
    sending_task = asyncio.create_task(auto_send_task())
    print(" Tâche d'envoi automatique démarrée (toutes les secondes)")


@app.on_event("shutdown")
async def shutdown_event():
    """Ferme le socket UDP et arrête les tâches à l'arrêt de l'API"""
    global sending_task, listener_task

    if listener_task:
        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass

    if sending_task:
        sending_task.cancel()
        try:
            await sending_task
        except asyncio.CancelledError:
            pass

    stop_udp()


@app.get("/")
async def root():
    """Page d'accueil avec statut"""
    return {
        "status": "running",
        "message": "Envoi automatique UDP avec Checksum simple",
        "port": dest_port,
        "connected_clients": len(connected_clients),
        "clients": [f"{ip}:{port}" for ip, port in connected_clients],
        "objects": STATIC_DATA["len"],
        "checksum_method": "Sum of all bytes (except header) & 0xFF",
    }


@app.get("/status")
async def status():
    """Retourne le statut de l'envoi automatique"""
    return {
        "udp_socket": "active" if sock else "inactive",
        "auto_send": (
            "active" if sending_task and not sending_task.done() else "inactive"
        ),
        "listener": (
            "active" if listener_task and not listener_task.done() else "inactive"
        ),
        "port": dest_port,
        "connected_clients": len(connected_clients),
        "clients": [f"{ip}:{port}" for ip, port in connected_clients],
        "interval": "1 second",
        "checksum": "Simple sum (excluding header) & 0xFF",
        "data": STATIC_DATA,
    }


@app.get("/clients")
async def get_clients():
    """Liste tous les clients connectés"""
    return {
        "count": len(connected_clients),
        "clients": [
            {"ip": ip, "port": port, "address": f"{ip}:{port}"}
            for ip, port in connected_clients
        ],
    }


@app.post("/clients/clear")
async def clear_clients():
    """Efface la liste des clients connectés"""
    global connected_clients
    count = len(connected_clients)
    connected_clients.clear()
    return {"message": f"{count} client(s) retiré(s)", "remaining": 0}


# Lancement du serveur
if __name__ == "__main__":
    print("🚀 Démarrage du serveur FastAPI...")
    print("📡 UDP Server avec clients enregistrés")
    print("✓ Checksum simple activé (somme sans header) & 0xFF")
    print("⏱ Envoi automatique toutes les secondes")
    print("🎧 Écoute des clients sur le port UDP")
    print("🌐 API disponible sur http://0.0.0.0:8000")
    print("📚 Documentation: http://0.0.0.0:8000/docs")
    print("📊 Statut: http://localhost:8000/status")
    print("👥 Clients: http://localhost:8000/clients")
    print(
        "\n💡 Pour qu'un client se connecte, il doit envoyer un message UDP au serveur"
    )

    uvicorn.run(app, host="0.0.0.0", port=8000)

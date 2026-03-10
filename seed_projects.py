import os
import sys

# Ensure the app module can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.db.database import SessionLocal, engine
from app.models.product import Product, Base
from app.models.user_story import UserStory

def seed_data():
    # Make sure all tables are created
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    
    print("Seeding project ideas and user stories...")

    # Data to seed
    projects_data = [
        {
            "name": "Nova HealthLens (AI Health Monitoring System)",
            "description": "A privacy-first wearable integration platform that uses on-device LLMs to analyze user biometrics and detect early signs of physiological anomalies.",
            "stories": [
                {
                    "title": "Analyze smartwatch data with on-device AI",
                    "description": "As a user, I want my smartwatch data parsed by an AI so that I get real-time health alerts without data leaving my device.\nTech Stack: Swift, CoreML, Edge TPU\nWhat to build: A background service capturing biometric data streams, running local inference, and pushing local notifications.\nFollow-up: As a user, I want a monthly health report PDF generated from my localized data.",
                    "tag": "Mobile/AI"
                },
                {
                    "title": "Doctor dashboard for encrypted summary reports",
                    "description": "As a doctor, I want to receive encrypted summary reports on my patient's dashboard so I can proactively schedule checkups.\nTech Stack: React, FHIR, Node.js\nWhat to build: A secure web portal displaying patient vital summaries and a REST API for ingesting encrypted reports.\nFollow-up: As a doctor, I want an alert for abnormal vital trends.",
                    "tag": "Web/Backend"
                }
            ]
        },
        {
            "name": "EcoTrace (Supply Chain Carbon Footprint Tracker)",
            "description": "A blockchain-backed ERP module that traces raw materials from source to shelf to calculate and display the real-time carbon footprint of each product.",
            "stories": [
                {
                    "title": "Scan shipment QR to log carbon data",
                    "description": "As a logistics manager, I want to scan a shipment QR code to instantly log its carbon emission data into the ledger.\nTech Stack: React Native, Hyperledger Fabric, Go\nWhat to build: A mobile app with a QR scanner that constructs a transaction payload and submits it to a smart contract.\nFollow-up: As a manager, I want to see a route-optimization suggestion to reduce emissions.",
                    "tag": "Mobile/Blockchain"
                },
                {
                    "title": "View carbon history on storefront app",
                    "description": "As a consumer, I want to view the carbon history of a product on the store app so I can make eco-friendly choices.\nTech Stack: Vue.js, GraphQL, PostgreSQL\nWhat to build: A GraphQL API querying aggregated ledger data and a Vue frontend component to visualize the footprint timeline.\nFollow-up: As a consumer, I want to compare two products' footprints side-by-side.",
                    "tag": "Frontend/API"
                }
            ]
        },
        {
            "name": "QuantumVault (Next-Gen Distributed Cloud Storage)",
            "description": "A highly secure, decentralized file storage system using quantum-resistant encryption algorithms for enterprise clients handling sensitive IP.",
            "stories": [
                {
                    "title": "Automated sharding and lattice encryption for huge datasets",
                    "description": "As an enterprise admin, I want to upload massive datasets that are automatically sharded and encrypted using lattice-based cryptography.\nTech Stack: Rust, WebAssembly, IPFS\nWhat to build: A WASM-based client utility for local encryption/sharding before uploading chunks to an IPFS network.\nFollow-up: As an admin, I want to set auto-expiry protocols for shared fragments.",
                    "tag": "Rust/Web3"
                },
                {
                    "title": "Immutable audit log of data fragment access",
                    "description": "As a compliance officer, I want an immutable audit log of who accessed which data shards and when.\nTech Stack: Next.js, FastAPI, ClickHouse\nWhat to build: A high-throughput telemetry ingestion endpoint and a Next.js admin dashboard to query access logs.\nFollow-up: As an officer, I want automated alerts for unauthorized decryption attempts.",
                    "tag": "Fullstack/Data"
                }
            ]
        },
        {
            "name": "OmniRetail (Immersive VR Commerce Platform)",
            "description": "A cross-platform virtual reality shopping mall where users can try on clothes using digital twins and purchase physical items that map to digital NFTs.",
            "stories": [
                {
                    "title": "Upload 3D scan for virtual apparel try-on",
                    "description": "As a shopper, I want to upload a 3D scan of myself so I can virtually try on apparel to ensure accurate sizing.\nTech Stack: Unity 3D, C#, Python OpenCV\nWhat to build: A Unity physics scene that fits a standardized clothing mesh over the uploaded user 3D body model.\nFollow-up: As a shopper, I want to see fabric physics simulate movement in real-time.",
                    "tag": "VR/CV"
                },
                {
                    "title": "Drag-and-drop 3D models into virtual storefront",
                    "description": "As a brand owner, I want to drag-and-drop 3D product models into my virtual storefront without writing code.\nTech Stack: Three.js, React, AWS S3\nWhat to build: A web-based CMS portal using Three.js to preview 3D assets before publishing them to the VR marketplace.\nFollow-up: As a brand owner, I want analytics on which items users interacted with longest.",
                    "tag": "3D/Web"
                }
            ]
        },
        {
            "name": "Nexus Grid (AI-Optimized Smart City Traffic Controller)",
            "description": "An intelligent traffic management system that ingests live CCTV, weather, and mobile data to adjust traffic light timings in real-time, minimizing congestion.",
            "stories": [
                {
                    "title": "Live traffic dashboard with predicted chokepoints",
                    "description": "As a city planner, I want a live dashboard showing current traffic flow and AI-predicted congestion chokepoints.\nTech Stack: Python, PyTorch, Apache Kafka\nWhat to build: A streaming pipeline feeding video data to a PyTorch prediction model, streaming results via websockets to the UI.\nFollow-up: As a planner, I want to simulate road closures and see predicted impact.",
                    "tag": "Data/AI"
                },
                {
                    "title": "Commuter notification for detected route anomalies",
                    "description": "As a commuter, I want an app notification advising me to leave 10 minutes earlier due to a detected anomaly on my route.\nTech Stack: Flutter, Firebase, Google Maps API\nWhat to build: A push notification service that triggers when the traffic AI identifies a severe backup on a user's saved route.\nFollow-up: As a commuter, I want alternative routing that specifically avoids emergency vehicle paths.",
                    "tag": "Mobile/Cloud"
                }
            ]
        }
    ]

    try:
        count = 0
        for p_data in projects_data:
            # Check if product already exists
            existing_product = db.query(Product).filter(Product.idea_name == p_data["name"]).first()
            if not existing_product:
                product = Product(idea_name=p_data["name"], description=p_data["description"])
                db.add(product)
                db.flush() # flush to get the ID
                
                for s_data in p_data["stories"]:
                    story = UserStory(
                        product_id=product.id,
                        title=s_data["title"],
                        description=s_data["description"],
                        tag=s_data["tag"]
                    )
                    db.add(story)
                count += 1
                
        db.commit()
        print(f"Successfully added {count} new project ideas and their user stories.")
    except Exception as e:
        db.rollback()
        print(f"Error seeding database: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_data()

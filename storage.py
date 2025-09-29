import json
import sqlite3
import os
import shutil
from datetime import datetime
from pathlib import Path
import streamlit as st

class BananaStorage:
    def __init__(self, storage_dir="banana_storage"):
        self.storage_dir = Path(storage_dir)
        self.images_dir = self.storage_dir / "images"
        self.json_file = self.storage_dir / "analyses.json"
        self.db_file = self.storage_dir / "banana_analyses.db"
        
        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self._init_json_storage()
        self._init_database()
    
    def _init_json_storage(self):
        """Initialize JSON storage file"""
        if not self.json_file.exists():
            with open(self.json_file, 'w') as f:
                json.dump({"analyses": [], "metadata": {"max_records": 10}}, f, indent=2)
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_filename TEXT UNIQUE,
                banana_count INTEGER,
                total_sugar_grams REAL,
                avg_ripeness_score REAL,
                total_age_days REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS banana_details (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER,
                banana_index INTEGER,
                age_days REAL,
                sugar_percentage REAL,
                sugar_grams REAL,
                ripeness_score REAL,
                green_percent REAL,
                yellow_percent REAL,
                brown_percent REAL,
                spot_count INTEGER,
                FOREIGN KEY (analysis_id) REFERENCES analyses (id) ON DELETE CASCADE
            )
        ''')
        
        # Create trigger to maintain 10 record limit
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS limit_analyses
            AFTER INSERT ON analyses
            BEGIN
                DELETE FROM analyses 
                WHERE id IN (
                    SELECT id FROM analyses 
                    ORDER BY timestamp DESC 
                    LIMIT -1 OFFSET 10
                );
            END;
        ''')
        
        conn.commit()
        conn.close()
    
    def save_image(self, image_data, filename):
        """Save image with FIFO management (max 10 images)"""
        # Ensure it's a JPEG file
        if not filename.lower().endswith(('.jpg', '.jpeg')):
            filename = f"{Path(filename).stem}.jpg"
        
        image_path = self.images_dir / filename
        
        # Check if we need to delete oldest image
        existing_images = list(self.images_dir.glob("*.jpg"))
        if len(existing_images) >= 10:
            # Find and delete oldest file
            oldest_image = min(existing_images, key=os.path.getctime)
            oldest_image.unlink()
        
        # Save new image
        with open(image_path, 'wb') as f:
            f.write(image_data)
        
        return filename
    
    def save_analysis(self, image_filename, detections, total_sugar, sugar_formula, weight_per_banana):
        """Save analysis to all storage systems"""
        timestamp = datetime.now().isoformat()
        
        # Prepare analysis data
        analysis_data = {
            "timestamp": timestamp,
            "image_filename": image_filename,
            "banana_count": len(detections),
            "total_sugar_grams": total_sugar,
            "sugar_formula": sugar_formula,
            "weight_per_banana": weight_per_banana,
            "detections": []
        }
        
        # Add detection details
        for i, det in enumerate(detections):
            analysis_data["detections"].append({
                "banana_index": i + 1,
                "age_days": det['analysis']['age_days'],
                "sugar_percentage": det.get('sugar_percentage', 0),
                "sugar_grams": det.get('sugar_content', 0),
                "ripeness_score": det['analysis']['ripeness_score'],
                "green_percent": det['analysis']['green_percent'],
                "yellow_percent": det['analysis']['yellow_percent'],
                "brown_percent": det['analysis']['brown_percent'],
                "spot_count": det['analysis']['spot_count']
            })
        
        # Save to JSON
        self._save_to_json(analysis_data)
        
        # Save to Database
        self._save_to_database(analysis_data)
        
        return analysis_data
    
    def _save_to_json(self, analysis_data):
        """Save analysis to JSON with 10-record limit"""
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        
        # Get the last inserted database ID to link JSON and DB records
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM analyses WHERE image_filename = ? ORDER BY id DESC LIMIT 1", 
                    (analysis_data["image_filename"],))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            analysis_data["db_id"] = result[0]
        
        data["analyses"].append(analysis_data)
        
        # Maintain 10 record limit
        if len(data["analyses"]) > 10:
            data["analyses"] = data["analyses"][-10:]
        
        with open(self.json_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_to_database(self, analysis_data):
        """Save analysis to SQLite database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        try:
            # Insert main analysis
            cursor.execute('''
                INSERT INTO analyses 
                (timestamp, image_filename, banana_count, total_sugar_grams, avg_ripeness_score, total_age_days)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                analysis_data["timestamp"],
                analysis_data["image_filename"],
                analysis_data["banana_count"],
                analysis_data["total_sugar_grams"],
                self._calculate_avg_ripeness(analysis_data["detections"]),
                self._calculate_total_age(analysis_data["detections"])
            ))
            
            analysis_id = cursor.lastrowid
            
            # Insert banana details
            for det in analysis_data["detections"]:
                cursor.execute('''
                    INSERT INTO banana_details 
                    (analysis_id, banana_index, age_days, sugar_percentage, sugar_grams, 
                     ripeness_score, green_percent, yellow_percent, brown_percent, spot_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    analysis_id,
                    det["banana_index"],
                    det["age_days"],
                    det["sugar_percentage"],
                    det["sugar_grams"],
                    det["ripeness_score"],
                    det["green_percent"],
                    det["yellow_percent"],
                    det["brown_percent"],
                    det["spot_count"]
                ))
            
            conn.commit()
            
        except sqlite3.IntegrityError:
            # Handle duplicate image filename
            st.warning("Analysis for this image already exists in database.")
        finally:
            conn.close()
    
    def _calculate_avg_ripeness(self, detections):
        """Calculate average ripeness score"""
        if not detections:
            return 0.0
        return sum(det['ripeness_score'] for det in detections) / len(detections)
    
    def _calculate_total_age(self, detections):
        """Calculate total age days"""
        return sum(det['age_days'] for det in detections)
    
    def get_storage_stats(self):
        """Get current storage statistics"""
        # JSON stats
        with open(self.json_file, 'r') as f:
            json_data = json.load(f)
        json_count = len(json_data["analyses"])
        
        # Database stats
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM analyses")
        db_count = cursor.fetchone()[0]
        conn.close()
        
        # Image stats
        image_count = len(list(self.images_dir.glob("*.jpg")))
        
        return {
            "json_analyses": json_count,
            "db_analyses": db_count,
            "images": image_count,
            "max_limit": 10
        }
    
    def get_recent_analyses(self, limit=5):
        """Get recent analyses for display"""
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        
        return data["analyses"][-limit:][::-1]  # Return most recent first
    
    def get_analysis_history(self):
        """Get full analysis history"""
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        
        return data["analyses"]
    
    def get_database_analyses(self):
        """Get analyses from database for advanced queries"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT a.*, COUNT(b.id) as banana_count
            FROM analyses a
            LEFT JOIN banana_details b ON a.id = b.analysis_id
            GROUP BY a.id
            ORDER BY a.timestamp DESC
        ''')
        
        columns = [desc[0] for desc in cursor.description]
        analyses = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return analyses
    
    def clear_storage(self):
        """Clear all storage (for testing/reset)"""
        # Clear JSON
        with open(self.json_file, 'w') as f:
            json.dump({"analyses": [], "metadata": {"max_records": 10}}, f, indent=2)
        
        # Clear database
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM banana_details")
        cursor.execute("DELETE FROM analyses")
        conn.commit()
        conn.close()
        
        # Clear images
        for image_file in self.images_dir.glob("*.jpg"):
            image_file.unlink()

    def delete_analysis_by_id(self, analysis_id):
        """Delete specific analysis by ID from all storage systems"""
        # Delete from database
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
        conn.commit()
        conn.close()
        
        # Delete from JSON
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        
        data["analyses"] = [a for a in data["analyses"] if a.get('db_id') != analysis_id]
        
        with open(self.json_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True

    def delete_oldest_analysis(self):
        """Delete the oldest analysis to make space"""
        # Get oldest analysis from database
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT id, image_filename FROM analyses ORDER BY timestamp ASC LIMIT 1")
        result = cursor.fetchone()
        
        if result:
            analysis_id, image_filename = result
            # Delete from database
            cursor.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
            conn.commit()
            conn.close()
            
            # Delete image file
            image_path = self.images_dir / image_filename
            if image_path.exists():
                image_path.unlink()
            
            # Delete from JSON
            with open(self.json_file, 'r') as f:
                data = json.load(f)
            
            data["analyses"] = [a for a in data["analyses"] if a.get('db_id') != analysis_id]
            
            with open(self.json_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            return True
        return False

    def get_all_analyses_with_ids(self):
        """Get all analyses with their database IDs"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT a.*, COUNT(b.id) as banana_count
            FROM analyses a
            LEFT JOIN banana_details b ON a.id = b.analysis_id
            GROUP BY a.id
            ORDER BY a.timestamp DESC
        ''')
        
        columns = [desc[0] for desc in cursor.description]
        analyses = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return analyses

    def delete_analysis_by_filename(self, filename):
        """Delete analysis by image filename"""
        # Delete image file
        image_path = self.images_dir / filename
        if image_path.exists():
            image_path.unlink()
        
        # Delete from database
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM analyses WHERE image_filename = ?", (filename,))
        conn.commit()
        conn.close()
        
        # Delete from JSON
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        
        data["analyses"] = [a for a in data["analyses"] if a.get('image_filename') != filename]
        
        with open(self.json_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    
    def export_data(self, export_path="banana_analysis_export.json"):
        """Export all data to a single JSON file"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "storage_stats": self.get_storage_stats(),
            "analyses": self.get_analysis_history(),
            "database_analyses": self.get_database_analyses()
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return export_path
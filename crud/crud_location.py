from sqlalchemy.orm import Session
from typing import List
from database import PredictedLocation
from schemas import PredictedLocationCreate, PredictedLocationUpdate

def get_predicted_location(db: Session, location_id: int) -> PredictedLocation:
    return db.query(PredictedLocation).filter(PredictedLocation.id == location_id).first()

def get_all_predicted_locations(db: Session, skip: int = 0, limit: int = 100) -> List[PredictedLocation]:
    return db.query(PredictedLocation).offset(skip).limit(limit).all()

def create_predicted_location(db: Session, location: PredictedLocationCreate) -> PredictedLocation:
    db_location = PredictedLocation(**location.model_dump())
    db.add(db_location)
    db.commit()
    db.refresh(db_location)
    return db_location

def update_predicted_location(db: Session, location_id: int, location_update: PredictedLocationUpdate) -> PredictedLocation:
    db_location = db.query(PredictedLocation).filter(PredictedLocation.id == location_id).first()
    if not db_location:
        return None
    update_data = location_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_location, key, value)
    db.commit()
    db.refresh(db_location)
    return db_location

def delete_predicted_location(db: Session, location_id: int) -> PredictedLocation:
    db_location = db.query(PredictedLocation).filter(PredictedLocation.id == location_id).first()
    if db_location:
        db.delete(db_location)
        db.commit()
        return db_location
    return None
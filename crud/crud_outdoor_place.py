from sqlalchemy.orm import Session
from typing import List, Optional
from database import models
import schemas

def get_place(db: Session, place_id: str) -> Optional[models.OutdoorPlace]:
    return db.query(models.OutdoorPlace).filter(models.OutdoorPlace.place_id == place_id).first()

def get_places(db: Session, skip: int = 0, limit: int = 100) -> List[models.OutdoorPlace]:
    return db.query(models.OutdoorPlace).offset(skip).limit(limit).all()

def create_place(db: Session, place: schemas.OutdoorPlaceCreate) -> models.OutdoorPlace:
    db_place = models.OutdoorPlace(**place.model_dump())
    db.add(db_place)
    db.commit()
    db.refresh(db_place)
    return db_place

def update_place(db: Session, place_id: str, place_update: schemas.OutdoorPlaceUpdate) -> Optional[models.OutdoorPlace]:
    db_place = get_place(db, place_id=place_id)
    if db_place:
        update_data = place_update.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(db_place, key, value)
        db.commit()
        db.refresh(db_place)
    return db_place

def delete_place(db: Session, place_id: str) -> Optional[models.OutdoorPlace]:
    db_place = get_place(db, place_id=place_id)
    if db_place:
        db.delete(db_place)
        db.commit()
    return db_place
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import crud
import schemas
from database import get_db

router = APIRouter()

@router.post("/", response_model=schemas.OutdoorPlace, status_code=status.HTTP_201_CREATED)
def create_new_place(place: schemas.OutdoorPlaceCreate, db: Session = Depends(get_db)):
    db_place = crud.get_place(db, place_id=place.place_id)
    if db_place:
        raise HTTPException(status_code=400, detail="Place ID already registered")
    return crud.create_place(db=db, place=place)

@router.get("/", response_model=List[schemas.OutdoorPlace])
def read_all_places(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    places = crud.get_places(db, skip=skip, limit=limit)
    return places

@router.get("/{place_id}", response_model=schemas.OutdoorPlace)
def read_one_place(place_id: str, db: Session = Depends(get_db)):
    db_place = crud.get_place(db, place_id=place_id)
    if db_place is None:
        raise HTTPException(status_code=404, detail="Place not found")
    return db_place

@router.patch("/{place_id}", response_model=schemas.OutdoorPlace)
def update_existing_place(place_id: str, place_update: schemas.OutdoorPlaceUpdate, db: Session = Depends(get_db)):
    db_place = crud.update_place(db, place_id=place_id, place_update=place_update)
    if db_place is None:
        raise HTTPException(status_code=404, detail="Place not found")
    return db_place

@router.delete("/{place_id}", response_model=schemas.OutdoorPlace)
def delete_existing_place(place_id: str, db: Session = Depends(get_db)):
    db_place = crud.delete_place(db, place_id=place_id)
    if db_place is None:
        raise HTTPException(status_code=404, detail="Place not found")
    return db_place
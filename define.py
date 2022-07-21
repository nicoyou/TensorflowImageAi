import enum

class ImageDataKey(str, enum.Enum):
	people = "people"

class PersonDataKey(str, enum.Enum):
	face_score = "face_score"
	face_pos = "face_pos"
	age = "age"
	
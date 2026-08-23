from typing import TypedDict

class Student(TypedDict):
    name: str
    age: int
    skill: str

student: Student = {
    'name': 'khan',
    'age': 20,
    'skill': 'AI'
}

print(student)
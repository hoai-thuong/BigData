## Getting Started
### Using
`Python :  Python 3.11.1 `

`Database: MongoDB`
### Create virtual enviroment (recommend)
```
python -m venv tutorial-env
```
#### On Mac
```
source tutorial-env/bin/activate
```
#### On Windows
```
tutorial-env\Scripts\activate
```
### Installation

> In app.py , config your Database name, and change name of class with your collection's name
 ``` 
 app.config['MONGODB_SETTINGS'] = {
    'db': 'your_new_database_name', 
    'host': 'localhost',
    'port': 27017} 
```
```
class bacsi(db.Document):  change bacsi = "your_collection_name"
    name = db.StringField()
    age = db.StringField()
    region = db.StringField()
    profile_pic = db.StringField()
    result = db.StringField()
```


```
pip install -r require.txt
```

## How to use

To run this application firstly execute `python3 app.py`, after which the `flask` built-in server would start hosting the application at localhost i.e.
`http://127.0.0.1:5000/`

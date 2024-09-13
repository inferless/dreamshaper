INPUT_SCHEMA = {
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["Image of tortoise painted in color"]
    },
   "input_image_url": {
    'datatype': 'STRING',
    'required': True,
    'shape': [1],
    'example': ["https://raahul-test-model-repo.s3.amazonaws.com/tortoise_1.png"]
    }
}

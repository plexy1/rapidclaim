const express = require('express');
const { PythonShell } = require('python-shell');

const app = express();
app.use(express.json());

app.post('/predict', (req, res) => {
  const features = req.body.features;
  PythonShell.run(
    '../backend/model/predict.py',
    { args: [JSON.stringify(features)] },
    (err, results) => {
      if (err) {
        res.status(500).json({ error: err.message });
        return;
      }
      res.json({ prediction: results[0] });
    }
  );
});

// app.get('/', (req, res) => {
//   res.send('RapidClaim backend is running. Use POST /predict.');
// });

app.listen(3000, () => console.log('Node server running at http://localhost:3000'));

const form = document.getElementById('avaliacao-risco');
const numericFields = [
    { el: document.getElementById('idade'), min: 0, max: 120, label: 'Idade' },
    { el: document.getElementById('nivel_glicose'), min: 0, max: 300, label: 'Nível de glicose' },
    { el: document.getElementById('imc'), min: 0, max: 100, label: 'IMC' }
];

function validateNumberField(field, min, max, label) {
    const value = field.value.trim();
    if (value === '') { field.setCustomValidity('Este campo é obrigatório.'); return; }
    const num = Number(value.replace(',', '.'));
    if (Number.isNaN(num)) { field.setCustomValidity('Insira um número válido.'); return; }
    if (num < min || num > max) { field.setCustomValidity(label + ' deve estar entre ' + min + ' e ' + max + '.'); return; }
    field.setCustomValidity('');
}

numericFields.forEach(({ el, min, max, label }) => {
    el.addEventListener('input', () => validateNumberField(el, min, max, label));
    el.addEventListener('blur', () => validateNumberField(el, min, max, label));
});

form.addEventListener('submit', (e) => {
    numericFields.forEach(({ el, min, max, label }) => validateNumberField(el, min, max, label));
    if (!form.checkValidity()) { e.preventDefault(); form.reportValidity(); }
});

// add class 'entered' to number fields if they have a value
document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.form-control input[type="number"]').forEach(function(el) {
    if (el.value) el.classList.add('entered');
    el.addEventListener('input', function() {
      if (this.value) this.classList.add('entered');
    });
  });
});
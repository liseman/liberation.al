import { describe, expect, it } from 'vitest';
import { unnamed, viewName } from './views';

describe('viewName', () => {
  it.each(['__proto__', 'constructor', 'toString'])('treats %s as an exact camera ID, with only its own alias', cam => {
    expect(viewName(cam, {})).toBe(cam);
    expect(viewName(cam, JSON.parse(`{"${cam}": "front door"}`))).toBe('Front door');
  });
  it("is the camera's name with a capital, or its id when it has none", () => {
    const views = { cam0: 'front camera', cam1: '  rear camera ' };
    expect(viewName('cam0', views)).toBe('Front camera');
    expect(viewName('cam1', views)).toBe('Rear camera');
    expect(viewName('cam2', views)).toBe('cam2');
    expect(viewName('cam0', { cam0: '   ' })).toBe('cam0');
  });
});

describe('unnamed', () => {
  it('is the cameras with no name or a blank one, in order', () => {
    expect(unnamed(['cam0', 'cam1', 'cam2'], { cam0: 'front', cam2: '  ' })).toEqual(['cam1', 'cam2']);
    expect(unnamed(['cam0'], { cam0: 'front' })).toEqual([]);
  });
  it('leaves out an id someone chose, which already reads as a name', () => {
    expect(unnamed(['patio', 'desk-left', 'camera-3', 'CAM_4'], {})).toEqual(['camera-3', 'CAM_4']);
  });
});

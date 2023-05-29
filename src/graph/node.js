import G6 from '@antv/g6';
import { foldString, latexToImg } from '@/utils/math';
import style from '@/stores/styles';
import system from '@/stores/system';

const drawRegular = (cfg, group) => {
  const rendered_content = latexToImg(cfg.content);
  const rendered_rules = cfg.rules.map((key) => latexToImg(key));
  const render = [rendered_content, ...rendered_rules];

  const mw = Math.max(
    Math.max(...render.map((item) => item.width)),
    style.min_width
  );
  const mh = Math.max(
    render.reduce((acc, item) => acc + item.height, 0) + style.m,
    style.min_height
  );

  // set neuron size to mw, mh
  const node_width = 2 * style.p + mw;
  const node_height = 2 * style.p + mh;
  cfg.size = [node_width, node_height];

  const start_x = -node_width / 2;
  const start_y = -node_height / 2;

  const shape = group.addShape('rect', {
    attrs: {
      x: start_x,
      y: start_y,
      width: node_width,
      height: node_height,
      stroke: system.dark ? style.darkContent : style.content,
      lineWidth: style.lineInactive,
      shadowColor: style.primary,
      shadowBlur: 0,
      radius: style.r,
      fill: system.dark ? style.dark : style.base,
    },
    name: 'neuron',
    draggable: true,
    zIndex: 10,
  });

  const content = group.addShape('image', {
    attrs: {
      y: start_y + style.p,
      x: start_x + (style.p + mw / 2 - render[0].width / 2),
      width: render[0].width,
      height: render[0].height,
      img: render[0].dom,
    },
    name: 'content',
    draggable: true,
    zIndex: 20,
  });

  cfg.rules.forEach((_, index) => {
    const rule = group.addShape('image', {
      attrs: {
        width: render[1 + index].width,
        height: render[1 + index].height,
        y:
          -node_height / 2 +
          style.p +
          style.m +
          render
            .slice(0, 1 + index)
            .reduce((acc, item) => acc + item.height, 0),
        x: -node_width / 2 + style.p + mw / 2 - render[1 + index].width / 2,
        img: render[1 + index].dom,
      },
      name: `rule-${index}`,
      visible: true,
      draggable: true,
      zIndex: 20,
    });
  });

  const ruleBoundary = group.addShape('rect', {
    attrs: {
      y: start_y + style.p + render[0].height + style.m,
      x: start_x + style.p,
      width: mw,
      height: render
        .slice(0, 1 + cfg.rules.length)
        .reduce((acc, item) => acc + item.height, 0),
      radius: 10,
      fill: '#00000000',
    },
    visible: true,
    name: 'rules',
    draggable: true,
    zIndex: 30,
  });

  const node_id = latexToImg(cfg.id);

  const id = group.addShape('image', {
    attrs: {
      x: start_x - 15 - node_id.width / 2,
      y: start_y - 15,
      width: node_id.width * 0.8,
      height: node_id.height * 0.8,
      img: node_id.dom,
    },
    name: 'label',
  });

  group.sort();

  return shape;
};

const drawInput = (cfg, group) => {
  const render = [latexToImg(foldString(`${cfg.content}`))];

  const mw = Math.max(
    Math.max(...render.map((item) => item.width)),
    style.min_width
  );
  const mh = Math.max(
    render.reduce((acc, item) => acc + item.height, 0) + style.m,
    style.min_height
  );

  // set neuron size to mw, mh
  const node_width = 2 * style.p + mw;
  const node_height = 2 * style.p + mh;
  cfg.size = [node_width, node_height];

  const start_x = -node_width / 2;
  const start_y = -node_height / 2;

  const shape = group.addShape('rect', {
    attrs: {
      x: start_x,
      y: start_y,
      width: node_width,
      height: node_height,
      stroke: system.dark ? style.darkContent : style.content,
      lineWidth: style.lineInactive,
      shadowColor: style.primary,
      shadowBlur: 0,
      radius: style.r,
      fill: system.dark ? style.dark : style.base,
    },
    name: 'neuron',
    draggable: true,
    zIndex: 10,
  });

  group.addShape('polygon', {
    attrs: {
      points: [
        [start_x + node_width / 2, start_y - 2],
        [start_x + node_width / 2 + 10, start_y - 15],
        [start_x + node_width / 2 - 10, start_y - 15],
      ],
      fill: style.base,
      stroke: system.dark ? style.darkContent : style.content,
      lineWidth: style.lineInactive,
    },
    name: 'input-indicator',
    draggable: true,
    zIndex: 10,
  });

  const content = group.addShape('image', {
    attrs: {
      y: -render[0].height / 2,
      x: start_x + (style.p + mw / 2 - render[0].width / 2),
      width: render[0].width,
      height: render[0].height,
      img: render[0].dom,
    },
    name: 'content',
    draggable: true,
    zIndex: 20,
  });

  const node_id = latexToImg(cfg.id);

  const id = group.addShape('image', {
    attrs: {
      x: start_x - 15 - node_id.width / 2,
      y: start_y - 15,
      width: node_id.width * 0.8,
      height: node_id.height * 0.8,
      img: node_id.dom,
    },
    name: 'label',
  });

  group.sort();

  return shape;
};

const drawOutput = (cfg, group) => {
  const render = [latexToImg(foldString(`${cfg.content}`))];

  const mw = Math.max(
    Math.max(...render.map((item) => item.width)),
    style.min_width
  );
  const mh = Math.max(
    render.reduce((acc, item) => acc + item.height, 0) + style.m,
    style.min_height
  );

  // set neuron size to mw, mh
  const node_width = 2 * style.p + mw;
  const node_height = 2 * style.p + mh;
  cfg.size = [node_width, node_height];

  const start_x = -node_width / 2;
  const start_y = -node_height / 2;

  const shape = group.addShape('rect', {
    attrs: {
      x: start_x,
      y: start_y,
      width: node_width,
      height: node_height,
      stroke: system.dark ? style.darkContent : style.content,
      lineWidth: style.lineInactive,
      shadowColor: style.primary,
      shadowBlur: 0,
      radius: style.r,
      fill: system.dark ? style.dark : style.base,
    },
    name: 'neuron',
    draggable: true,
    zIndex: 10,
  });

  group.addShape('rect', {
    attrs: {
      x: start_x + 5,
      y: start_y + 5,
      width: node_width - 10,
      height: node_height - 10,
      stroke: system.dark ? style.darkContent : style.content,
      lineWidth: style.lineInactive,
      shadowColor: style.primary,
      shadowBlur: 0,
      radius: style.r - 5,
      fill: system.dark ? style.dark : style.base,
    },
    name: 'output-indicator',
    draggable: true,
    zIndex: 11,
  });

  const content = group.addShape('image', {
    attrs: {
      y: -render[0].height / 2,
      x: start_x + (style.p + mw / 2 - render[0].width / 2),
      width: render[0].width,
      height: render[0].height,
      img: render[0].dom,
    },
    name: 'content',
    draggable: true,
    zIndex: 20,
  });

  const node_id = latexToImg(cfg.id);

  const id = group.addShape('image', {
    attrs: {
      x: start_x - 15 - node_id.width / 2,
      y: start_y - 15,
      width: node_id.width * 0.8,
      height: node_id.height * 0.8,
      img: node_id.dom,
    },
    name: 'label',
  });

  group.sort();

  return shape;
};

const setStateRegular = (name, value, item) => {
  const shape = item.get('keyShape');
  const { type, duration } = item.getModel();
  const rules = item.getContainer().findAll((ele) => {
    return ele.get('name')?.includes('rule');
  });
  const animations = item.getContainer().findAll((ele) => {
    return ele.get('name')?.includes('animation');
  });
  const content = item.getContainer().find((ele) => {
    return ele.get('name') === 'content';
  });
  // get original style of item
  const original_style = item._cfg.originStyle;

  if (name === 'simple') {
    content.attr('x', value ? -content.attr('width') / 2 : -60);
    content.attr('y', value ? -content.attr('height') / 2 : -60);

    shape.attr(
      'width',
      value
        ? Math.max(style.p + content.attr('width') + style.p, style.min_height)
        : 2 * 20 + 100
    );
    shape.attr('height', value ? style.min_height : 2 * 20 + 100);
    shape.attr('radius', value ? style.r / 2 : style.r);

    // recenter the shape
    shape.attr('x', -shape.attr('width') / 2);
    shape.attr('y', -shape.attr('height') / 2);

    if (type === 'input') {
      const indicator = item.getContainer().find((ele) => {
        return ele.get('name') === 'input-indicator';
      });
      indicator.attr(
        'points',
        value
          ? [
              [shape.attr('x') + shape.attr('width') / 2, shape.attr('y') - 2],
              [
                shape.attr('x') + shape.attr('width') / 2 + 10,
                shape.attr('y') - 15,
              ],
              [
                shape.attr('x') + shape.attr('width') / 2 - 10,
                shape.attr('y') - 15,
              ],
            ]
          : []
      );
    }

    if (type === 'output') {
      const indicator = item.getContainer().find((ele) => {
        return ele.get('name') === 'output-indicator';
      });
      indicator.attr('radius', value ? style.r / 2 + 5 : style.r + 5);
      indicator.attr('x', shape.attr('x') - 5);
      indicator.attr('y', shape.attr('y') - 5);
      indicator.attr('width', shape.attr('width') + 10);
      indicator.attr('height', shape.attr('height') + 10);
    }

    // recenter the label
    const label = item.getContainer().find((ele) => {
      return ele.get('name') === 'label';
    });

    label.attr('x', -shape.attr('width') / 2 - 15);
    label.attr('y', -shape.attr('height') / 2 - 15);

    rules.forEach((rule) => {
      value ? rule.hide() : rule.show();
    });
  }
  if (name === 'spiking') {
    shape.attr('stroke', value ? style.primary : style.content);
    shape.attr('lineWidth', value ? 5 : 2);
    shape.attr('shadowBlur', value ? 10 : 0);
  } else if (!['spiking', 'simple'].includes(name)) {
    const shapes = item.getStateStyle(name);

    if (!shapes) return;

    Object.keys(shapes)?.forEach((shapeName) => {
      const shapeItem = item
        .get('group')
        .find((e) => e.get('name') === shapeName);

      const attrs = shapes[shapeName];
      Object.keys(attrs).forEach((attr) => {
        const attr_value = shapes[shapeName][attr];
        const orig_value = original_style[shapeName][attr];
        shapeItem?.attr(attr, value ? attr_value : orig_value);
      });
    });
  }
};

const options = {
  stateStyles: {
    hover: {
      neuron: {
        shadowColor: style.primary,
        shadowBlur: 10,
      },
    },
    selected: {
      neuron: {
        shadowBlur: 10,
        lineWidth: style.lineActive,
        stroke: style.primary,
        shadowColor: style.primary,
      },
    },
    closed: {
      neuron: {
        shadowBlur: 10,
        stroke: style.error,
        shadowColor: style.error,
      },
    },
    default: {
      neuron: {
        shadowBlur: 0,
        shadowColor: style.primary,
        lineWidth: style.lineInactive,
        stroke: system.dark ? style.darkContent : style.content,
      },
    },
  },
};

export default function initializeNode() {
  G6.registerNode('regular', {
    options,
    drawShape: (cfg, group) => {
      return drawRegular(cfg, group);
    },
    setState: (name, value, item) => {
      return setStateRegular(name, value, item);
    },
  });
  G6.registerNode('output', {
    options,
    drawShape: (cfg, group) => {
      return drawOutput(cfg, group);
    },
    setState: (name, value, item) => {
      return setStateRegular(name, value, item);
    },
  });
  G6.registerNode('input', {
    options,
    drawShape: (cfg, group) => {
      return drawInput(cfg, group);
    },
    setState: (name, value, item) => {
      return setStateRegular(name, value, item);
    },
  });
}
